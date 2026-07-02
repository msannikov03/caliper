import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { applyNodeChanges, applyEdgeChanges, addEdge } from "@xyflow/react";
import type { Connection, NodeChange, EdgeChange } from "@xyflow/react";
import type {
  CNode,
  CEdge,
  GraphScope,
  Diagnostics,
  GraphRunResult,
  NodeStatus,
} from "./graph/types";
import { diagnosticsOk } from "./graph/types";
import type { KindName } from "./graph/spec";
import { defaultParams, outPortType, inPortTypes, PORT_COLORS, NODE_SPECS } from "./graph/spec";
import { serializeGraph, parseGraph } from "./graph/serialize";

// ---- wire types: mirror the serde structs in src-tauri/src/lib.rs exactly ----

export type JointKind = "revolute" | "prismatic";

export interface FrameInfo {
  name: string;
  /** index into frames[] of the parent frame to draw a rod to, or -1 (root). */
  parent: number;
  /** movable joint whose chain this frame rides on, or -1 (root/world). */
  anchor: number;
  /** if this frame is a joint's primary output frame: that joint index, else -1. */
  jointIndex: number;
  /** "revolute" | "prismatic" for a primary joint frame, else null. */
  jointKind: JointKind | null;
  /** joint-local axis for a primary joint frame, else null. */
  axis: [number, number, number] | null;
}

export type VisualKind = "box" | "sphere" | "cylinder" | "capsule" | "mesh";

/** One render-only URDF `<visual>`, flattened (mirrors `VisualDto` in lib.rs).
 *  `kind` selects the populated size fields: box → halfExtents; sphere → radius;
 *  cylinder/capsule → radius+length (Z-aligned, URDF convention); mesh →
 *  meshPath (absolute, null = unresolvable) + meshScale + raw. */
export interface VisualInfo {
  /** index into frames[]; world pose = frames[frame] · origin. */
  frame: number;
  /** shape-local offset within the frame, column-major 4x4. */
  origin: number[];
  kind: VisualKind;
  halfExtents: [number, number, number] | null;
  radius: number | null;
  length: number | null;
  /** URDF material RGBA in [0,1], else null (renderer picks a neutral tone). */
  color: [number, number, number, number] | null;
  meshPath: string | null;
  meshScale: [number, number, number] | null;
  /** raw URDF filename attribute (diagnostics for unresolved meshes). */
  raw: string | null;
}

export interface RobotInfo {
  name: string;
  ndof: number;
  jointNames: string[];
  jointKinds: JointKind[]; // parallel to jointNames
  limits: ([number, number] | null)[]; // parallel to jointNames
  frames: FrameInfo[];
  tip: number; // index into frames
  hasInertia: boolean; // dynamics available (every link has <inertial>)
  /** render-only <visual> geometry; empty/absent → the UI draws the rod skeleton. */
  visuals?: VisualInfo[];
}

/** column-major 4x4 (THREE.Matrix4 element order), exactly what fromArray wants. */
export type Mat4 = number[]; // length 16

export interface IkSolution {
  success: boolean;
  q: number[];
  residual: number;
}

export interface SingularityReport {
  manipulability: number;
  conditionNumber: number | null; // null == ∞
  sigmaMin: number;
  sigma: [number, number, number];
  kind: "none" | "wrist" | "elbow" | "boundary";
  offendingJoints: number[];
  epsActivate: number;
  tipWorld: [number, number, number];
  ellipsoidAxes: [[number, number, number], [number, number, number], [number, number, number]];
  ellipsoidRadii: [number, number, number];
}

export interface TrajectoryDto {
  kind: string;
  duration: number;
  ndof: number;
  dt: number;
  times: number[];
  q: number[][];
  qd: number[][];
  tipPath: [number, number, number][];
  frames: Mat4[][]; // N x nframes col-major — baked, playback is render-only
  ok: boolean;
  reached: number;
  maxJerkRatio: number;
}
export interface SimTrajectoryDto extends TrajectoryDto {
  energy: number[];
  energyDrift: number;
  settled: boolean;
  gravity: [number, number, number];
  damping: number;
}
export interface CollisionDto {
  collision: boolean;
  collidingFrames: string[];
  selfPairs: [string, string][];
  worldHits: string[];
  numColliders: number;
  uncoveredFrames: number;
}
export type StudioMode = "jog" | "motion" | "simulate" | "graph";
export interface NamedPoseDto {
  name: string;
  q: number[];
}

export interface StudioState {
  // robot + configuration
  robot: RobotInfo | null;
  q: number[]; // length ndof — the single source of truth for configuration
  frames: Mat4[]; // length frames.length — world poses, recomputed by the engine
  report: SingularityReport | null; // singularity/manipulability at the current q

  // status
  loading: boolean;
  error: string | null;
  ikOk: boolean | null;
  ikResidual: number | null;

  // internal: monotonic request id so older FK replies can't clobber newer ones
  _reqId: number;
  // independent guard for analyze() so a slow SVD can't gate the FK frame
  _analyzeReqId: number;

  // recent URDFs opened from disk (persisted to localStorage; sample fixtures excluded)
  recentUrdfs: string[];
  addRecent: (path: string) => void;
  removeRecent: (path: string) => void;
  loadRecent: () => void;

  // actions
  loadRobot: (path: string) => Promise<void>;
  setJoint: (i: number, v: number) => void;
  refreshFrames: () => Promise<void>;
  refreshAnalysis: () => Promise<void>;
  solveIkGoverned: (targetColMajor: Mat4, snap: boolean) => Promise<void>;

  // motion (Phase 3)
  traj: TrajectoryDto | null;
  poses: NamedPoseDto[];
  playing: boolean;
  playhead: number; // seconds
  planMoveJ: (qGoal: number[]) => Promise<void>;
  planMoveL: (targetColMajor: Mat4, frame?: string) => Promise<void>;
  planMoveToPose: (name: string) => Promise<void>;
  clearTraj: () => void;
  play: () => void;
  pause: () => void;
  seek: (t: number) => void;
  savePose: (name: string) => Promise<void>;
  deletePose: (name: string) => Promise<void>;
  refreshPoses: () => Promise<void>;
  _applyTrajAt: (t: number) => void;

  // simulation (Phase 4)
  mode: StudioMode;
  setMode: (m: StudioMode) => void;
  simTraj: SimTrajectoryDto | null;
  simGravity: boolean;
  simDamping: number;
  simTorque: number[];
  runGravityDrop: () => Promise<void>;
  clearSim: () => void;

  // control + collision (Phase 5)
  collision: CollisionDto | null;
  runControl: (goal: number[]) => Promise<void>;
  runPlan: (goal: number[]) => Promise<void>;
  checkCollision: (ground?: number | null) => Promise<void>;
  clearCollision: () => void;

  // graph mode (Phase 8) — Simulink-style dataflow editor
  graphNodes: CNode[];
  graphEdges: CEdge[];
  graphScopes: GraphScope[]; // last run's extracted Scope series
  graphLive: boolean; // when true, Scope charts stream the last run in (live feel)
  graphBanner: string | null; // red banner: cycles / type mismatch / run error
  graphSaved: string[]; // saved graph names (list_graphs)
  graphName: string;
  _graphRunId: number; // monotonic latest-wins guard, like _reqId
  onGraphNodesChange: (changes: NodeChange<CNode>[]) => void;
  onGraphEdgesChange: (changes: EdgeChange<CEdge>[]) => void;
  onGraphConnect: (c: Connection) => void;
  addGraphNode: (kind: KindName) => void;
  updateNodeParams: (id: string, patch: Record<string, unknown>) => void;
  setGraphName: (s: string) => void;
  runGraph: () => Promise<void>;
  runGraphLive: () => Promise<void>; // same run, but stream the Scope series in
  _execGraph: (live: boolean) => Promise<void>; // shared run impl (latest-wins)
  validateGraph: () => Promise<void>;
  saveGraph: (name: string) => Promise<void>;
  loadGraph: (name: string) => Promise<void>;
  refreshGraphList: () => Promise<void>;
}

export const useStore = create<StudioState>((set, get) => ({
  robot: null,
  q: [],
  frames: [],
  report: null,
  loading: false,
  error: null,
  ikOk: null,
  ikResidual: null,
  _reqId: 0,
  _analyzeReqId: 0,
  traj: null,
  poses: [],
  playing: false,
  playhead: 0,
  mode: "jog",
  simTraj: null,
  simGravity: true,
  simDamping: 0.2,
  simTorque: [],
  collision: null,
  graphNodes: [],
  graphEdges: [],
  graphScopes: [],
  graphLive: false,
  graphBanner: null,
  graphSaved: [],
  graphName: "",
  _graphRunId: 0,
  recentUrdfs: [],

  addRecent(path) {
    const recentUrdfs = mergeRecent(get().recentUrdfs, path);
    set({ recentUrdfs });
    persistRecents(recentUrdfs);
  },
  removeRecent(path) {
    const recentUrdfs = get().recentUrdfs.filter((p) => p !== path);
    set({ recentUrdfs });
    persistRecents(recentUrdfs);
  },
  loadRecent() {
    let stored: unknown = [];
    try {
      stored = JSON.parse(localStorage.getItem(RECENT_KEY) ?? "[]");
    } catch {
      stored = [];
    }
    const recentUrdfs = Array.isArray(stored)
      ? stored.filter((p): p is string => typeof p === "string").slice(0, RECENT_CAP)
      : [];
    set({ recentUrdfs });
    // prune entries whose file no longer exists (async; keeps the list honest)
    void pruneMissingRecents(get, set);
  },

  async loadRobot(path) {
    set({ loading: true, error: null });
    try {
      const robot = await invoke<RobotInfo>("robot_info", { path });
      stopClock();
      set({
        robot,
        q: new Array(robot.ndof).fill(0),
        frames: [],
        report: null,
        ikOk: null,
        ikResidual: null,
        traj: null,
        simTraj: null,
        mode: "jog",
        playing: false,
        playhead: 0,
        poses: [],
        // a new robot invalidates the (ndof-bound) graph
        graphNodes: [],
        graphEdges: [],
        graphScopes: [],
        graphLive: false,
        graphBanner: null,
        graphName: "",
      });
      await get().refreshFrames();
      await get().refreshPoses();
      void get().refreshGraphList();
    } catch (e) {
      set({ error: String(e), robot: null });
    } finally {
      set({ loading: false });
    }
  },

  setJoint(i, v) {
    if (get().playing || get().mode === "simulate" || get().mode === "graph") return; // sim/playback/graph own the pose
    const q = get().q.slice();
    q[i] = v;
    // a manual jog invalidates any prior IK status (it described a different pose)
    set({ q, ikOk: null, ikResidual: null });
    scheduleRefresh(get);
  },

  async refreshFrames() {
    const { q, robot } = get();
    if (!robot) return;
    const reqId = get()._reqId + 1;
    set({ _reqId: reqId });
    try {
      const frames = await invoke<Mat4[]>("get_frames", { q });
      if (get()._reqId !== reqId) return; // a newer request superseded us
      set({ frames, error: null }); // success clears any prior transient error
      void get().refreshAnalysis(); // ride each FK refresh → HUD + ellipsoid follow q
    } catch (e) {
      set({ error: String(e) });
    }
  },

  async refreshAnalysis() {
    const { q, robot } = get();
    if (!robot) return;
    const reqId = get()._analyzeReqId + 1;
    set({ _analyzeReqId: reqId });
    try {
      const report = await invoke<SingularityReport>("analyze", { q });
      if (get()._analyzeReqId !== reqId) return; // a newer analyze superseded us
      set({ report });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  async solveIkGoverned(targetColMajor, snap) {
    const { q, robot } = get();
    if (!robot || get().playing || get().mode === "simulate" || get().mode === "graph") return; // gizmo inert
    const frameName = robot.frames[robot.tip].name;
    const reqId = get()._reqId + 1;
    set({ _reqId: reqId });
    try {
      let res = await invoke<IkSolution>("solve_ik_governed", {
        req: { target: targetColMajor, seed: q, frame: frameName },
      });
      if (get()._reqId !== reqId) return; // out-of-order drag reply (skip the snap too)
      if (snap && res.success) {
        res = await invoke<IkSolution>("solve_ik", {
          req: { target: targetColMajor, seed: res.q, frame: frameName },
        });
        if (get()._reqId !== reqId) return; // superseded while snapping
      }
      set({ q: res.q, ikOk: res.success, ikResidual: res.residual, error: null });
      await get().refreshFrames(); // -> refreshAnalysis -> HUD + ellipsoid update live
    } catch (e) {
      set({ error: String(e) });
    }
  },

  // ---- motion (Phase 3) ----
  async planMoveJ(qGoal) {
    const { q, robot } = get();
    if (!robot) return;
    try {
      const traj = await invoke<TrajectoryDto>("plan_move_j", { req: { qStart: q, qGoal } });
      set({ traj, playhead: 0, playing: false });
      get()._applyTrajAt(0);
    } catch (e) {
      set({ error: String(e) });
    }
  },
  async planMoveL(targetColMajor, frame) {
    const { q, robot } = get();
    if (!robot) return;
    const frameName = frame ?? robot.frames[robot.tip].name;
    try {
      const traj = await invoke<TrajectoryDto>("plan_move_l", {
        req: { qStart: q, target: targetColMajor, frame: frameName },
      });
      set({ traj, playhead: 0, playing: false });
      get()._applyTrajAt(0);
    } catch (e) {
      set({ error: String(e) });
    }
  },
  async planMoveToPose(name) {
    const { q, robot } = get();
    if (!robot || get().mode === "simulate") return; // motion-only; sim owns the pose
    try {
      const traj = await invoke<TrajectoryDto>("plan_move_to_pose", { req: { qStart: q, name } });
      set({ traj, playhead: 0, playing: false });
      get()._applyTrajAt(0);
      get().play();
    } catch (e) {
      set({ error: String(e) });
    }
  },
  clearTraj() {
    stopClock();
    set({ traj: null, simTraj: null, playing: false, playhead: 0 });
  },
  play() {
    const traj = activeClip(get());
    if (!traj) return;
    base = get().playhead >= traj.duration ? 0 : get().playhead;
    t0 = performance.now();
    set({ playing: true });
    if (!rafId) rafId = requestAnimationFrame(() => tick(get, set));
  },
  pause() {
    stopClock();
    set({ playing: false });
  },
  seek(t) {
    const traj = activeClip(get());
    if (!traj) return;
    const tt = Math.max(0, Math.min(t, traj.duration));
    stopClock();
    set({ playhead: tt, playing: false });
    get()._applyTrajAt(tt);
  },
  async savePose(name) {
    const { q, robot } = get();
    if (!robot) return;
    try {
      await invoke("save_pose", { name, q });
      await get().refreshPoses();
    } catch (e) {
      set({ error: String(e) });
    }
  },
  async deletePose(name) {
    try {
      await invoke("delete_pose", { name });
      await get().refreshPoses();
    } catch (e) {
      set({ error: String(e) });
    }
  },
  async refreshPoses() {
    if (!get().robot) return;
    try {
      const poses = await invoke<NamedPoseDto[]>("list_poses");
      set({ poses });
    } catch (e) {
      set({ error: String(e) });
    }
  },
  _applyTrajAt(t) {
    const traj = activeClip(get());
    if (!traj) return;
    // pick the nearest baked frame row (render-only; no engine round-trip)
    let k = Math.round(t / traj.dt);
    k = Math.max(0, Math.min(k, traj.frames.length - 1));
    // bump _reqId so any in-flight get_frames/solve_ik reply can't clobber this pose
    set({ q: traj.q[k], frames: traj.frames[k], _reqId: get()._reqId + 1 });
    // throttle analysis: only re-run when the baked frame actually changes (or the
    // clip itself changed), instead of every playback rAF tick.
    if (k !== lastAnalyzedK || traj !== lastAnalyzedClip) {
      lastAnalyzedK = k;
      lastAnalyzedClip = traj;
      void get().refreshAnalysis(); // HUD/ellipsoid follow; latest-wins via _analyzeReqId
    }
  },

  // ---- simulation (Phase 4) ----
  setMode(m) {
    stopClock();
    set({ mode: m, traj: null, simTraj: null, playing: false, playhead: 0 });
    void get().refreshFrames();
  },
  async runGravityDrop() {
    const { q, robot, simGravity, simDamping, simTorque } = get();
    if (!robot) return;
    if (!robot.hasInertia) {
      set({ error: "this robot has no inertial data" });
      return;
    }
    const tau = simTorque.length === robot.ndof ? simTorque : new Array(robot.ndof).fill(0);
    try {
      const simTraj = await invoke<SimTrajectoryDto>("sim_drop", {
        req: {
          qStart: q,
          tau,
          gravity: simGravity ? [0, 0, -9.81] : [0, 0, 0],
          damping: simDamping,
          duration: 4.0,
          settle: true,
        },
      });
      set({ simTraj, traj: null, playhead: 0, playing: false });
      get()._applyTrajAt(0);
      get().play();
    } catch (e) {
      set({ error: String(e) });
    }
  },
  clearSim() {
    stopClock();
    set({ simTraj: null, playing: false, playhead: 0 });
    void get().refreshFrames();
  },

  // ---- control + collision (Phase 5) ----
  async runControl(goal) {
    const { q, robot } = get();
    if (!robot) return;
    if (!robot.hasInertia) {
      set({ error: "this robot has no inertial data" });
      return;
    }
    if (goal.length !== robot.ndof) {
      set({ error: `goal needs ${robot.ndof} values` });
      return;
    }
    try {
      const simTraj = await invoke<SimTrajectoryDto>("control_run", {
        req: { qStart: q, goal, duration: 4.0 },
      });
      set({ simTraj, traj: null, playhead: 0, playing: false });
      get()._applyTrajAt(0);
      get().play();
    } catch (e) {
      set({ error: String(e) });
    }
  },
  async runPlan(goal) {
    const { q, robot } = get();
    if (!robot) return;
    if (goal.length !== robot.ndof) {
      set({ error: `goal needs ${robot.ndof} values` });
      return;
    }
    try {
      const simTraj = await invoke<SimTrajectoryDto>("plan_run", {
        req: { qStart: q, goal },
      });
      set({ simTraj, traj: null, playhead: 0, playing: false });
      get()._applyTrajAt(0);
      get().play();
    } catch (e) {
      set({ error: String(e) });
    }
  },
  async checkCollision(ground = null) {
    const { q, robot } = get();
    if (!robot) return;
    try {
      const collision = await invoke<CollisionDto>("check_collision", {
        req: { q, ground },
      });
      set({ collision });
    } catch (e) {
      set({ error: String(e) });
    }
  },
  clearCollision() {
    set({ collision: null });
  },

  // ---- graph mode (Phase 8) ----
  onGraphNodesChange(changes) {
    set({ graphNodes: applyNodeChanges<CNode>(changes, get().graphNodes) });
  },
  onGraphEdgesChange(changes) {
    set({ graphEdges: applyEdgeChanges<CEdge>(changes, get().graphEdges) });
  },
  onGraphConnect(c) {
    const { source, target, sourceHandle, targetHandle } = c;
    if (!source || !target) return;
    const nodes = get().graphNodes;
    const src = nodes.find((n) => n.id === source);
    const tgt = nodes.find((n) => n.id === target);
    if (!src || !tgt) return;
    const outName = sourceHandle ?? NODE_SPECS[src.data.kind].outputs[0]?.name;
    const inName = targetHandle ?? NODE_SPECS[tgt.data.kind].inputs[0]?.name;
    if (!outName || !inName) return;
    const st = outPortType(src.data.kind, outName);
    const tts = inPortTypes(tgt.data.kind, inName);
    if (!st || !tts || !tts.includes(st)) {
      // reject type-incompatible wires (handle port-type guard)
      set({ graphBanner: `Incompatible wire: ${st ?? "?"} → ${tts ? tts.join("|") : "?"}` });
      return;
    }
    const color = PORT_COLORS[st];
    // reject back-edges: would adding source→target create a cycle?
    if (canReach(target, source, get().graphEdges)) {
      set({ graphBanner: `Rejected: ${source} → ${target} would create a cycle` });
      return;
    }
    // one feeder per input port: drop any existing edge into the same target handle
    const kept = get().graphEdges.filter(
      (e) => !(e.target === target && e.targetHandle === inName),
    );
    const edge: CEdge = {
      id: `e_${source}.${outName}->${target}.${inName}`,
      source,
      target,
      sourceHandle: outName,
      targetHandle: inName,
      data: { color },
      style: { stroke: color },
    };
    set({ graphEdges: addEdge<CEdge>(edge, kept), graphBanner: null });
  },
  addGraphNode(kind) {
    const ndof = get().robot?.ndof ?? 0;
    const count = get().graphNodes.length;
    const node: CNode = {
      id: nextNodeId(kind),
      type: kind,
      position: { x: 90 + (count % 5) * 64, y: 70 + (count % 7) * 46 },
      data: { kind, params: defaultParams(kind, ndof), status: "idle" },
    };
    set({ graphNodes: [...get().graphNodes, node] });
  },
  updateNodeParams(id, patch) {
    set({
      graphNodes: get().graphNodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, params: { ...n.data.params, ...patch } } } : n,
      ),
    });
  },
  setGraphName(s) {
    set({ graphName: s });
  },
  runGraph() {
    return get()._execGraph(false);
  },
  runGraphLive() {
    return get()._execGraph(true);
  },
  async _execGraph(live) {
    const robot = get().robot;
    if (!robot) return;
    const id = get()._graphRunId + 1;
    set({
      _graphRunId: id,
      graphBanner: null,
      // clear any prior live state up-front so a new batch run renders statically
      graphLive: false,
      graphNodes: get().graphNodes.map((n) => ({
        ...n,
        data: { ...n.data, status: "running" as NodeStatus, error: undefined },
      })),
    });
    const graphJson = serializeGraph(get().graphNodes, get().graphEdges, get().graphName, robot.name);
    try {
      const res = await invoke<GraphRunResult>("graph_run", { graphJson });
      if (get()._graphRunId !== id) return; // a newer run superseded us
      const ran = new Set(res.diagnostics?.topoOrder ?? []);
      set({
        graphScopes: res.scopes ?? [],
        // flip live on AFTER scopes land so the charts mount with fresh data and
        // their reveal effect re-runs in streaming mode
        graphLive: live && (res.scopes?.length ?? 0) > 0,
        graphNodes: get().graphNodes.map((n) => ({
          ...n,
          data: {
            ...n.data,
            status: (ran.size === 0 || ran.has(n.id) ? "ok" : "idle") as NodeStatus,
            error: undefined,
          },
        })),
      });
      const traj = res.trajectory ?? null;
      if (traj) {
        // the View sink drives the persistent GL preview via the playback clock
        set({ traj, simTraj: null, playhead: 0, playing: false });
        get()._applyTrajAt(0);
        get().play();
      } else {
        // no View sink in this graph — stop any stale clip from a previous run
        stopClock();
        set({ traj: null, simTraj: null, playing: false, playhead: 0 });
      }
    } catch (e) {
      if (get()._graphRunId !== id) return;
      handleGraphError(e, get, set);
    }
  },
  async validateGraph() {
    const robot = get().robot;
    if (!robot) return;
    const graphJson = serializeGraph(get().graphNodes, get().graphEdges, get().graphName, robot.name);
    try {
      const diag = await invoke<Diagnostics>("graph_validate", { graphJson });
      const dec = decorateWithDiagnostics(get().graphNodes, get().graphEdges, diag);
      set({
        graphNodes: dec.nodes,
        graphEdges: dec.edges,
        graphBanner: diagnosticsOk(diag) ? "graph is valid ✓" : dec.banner,
      });
    } catch (e) {
      set({ graphBanner: String(e) });
    }
  },
  async saveGraph(name) {
    const robot = get().robot;
    if (!robot) return;
    const graphJson = serializeGraph(get().graphNodes, get().graphEdges, name, robot.name);
    try {
      await invoke("save_graph", { name, graphJson });
      set({ graphName: name });
      await get().refreshGraphList();
    } catch (e) {
      set({ graphBanner: String(e) });
    }
  },
  async loadGraph(name) {
    const ndof = get().robot?.ndof ?? 0;
    try {
      const graphJson = await invoke<string>("load_graph", { name });
      const parsed = parseGraph(graphJson, ndof);
      // re-color edges by their source out-port type (parseGraph leaves them bare)
      const colored = parsed.edges.map((e) => {
        const src = parsed.nodes.find((n) => n.id === e.source);
        const t = src && e.sourceHandle ? outPortType(src.data.kind, e.sourceHandle) : undefined;
        const color = t ? PORT_COLORS[t] : "#6c6c7a";
        return { ...e, data: { color }, style: { stroke: color } };
      });
      bumpNodeSeq(parsed.nodes);
      set({
        graphNodes: parsed.nodes,
        graphEdges: colored,
        graphName: parsed.name || name,
        graphScopes: [],
        graphLive: false,
        graphBanner: null,
      });
    } catch (e) {
      set({ graphBanner: String(e) });
    }
  },
  async refreshGraphList() {
    try {
      const list = await invoke<string[]>("list_graphs");
      set({ graphSaved: list });
    } catch {
      // the backend command may be absent in some builds; leave the list as-is.
    }
  },
}));

/// The active playback clip: a baked sim rollout takes precedence over a motion traj.
function activeClip(s: StudioState): TrajectoryDto | null {
  return s.simTraj ?? s.traj;
}

// ---- graph helpers (module scope) ----

/// BFS reachability: returns true if `startId` can reach `goalId` via `edges`.
/// Used to detect cycles before accepting a new connection.
function canReach(startId: string, goalId: string, edges: CEdge[]): boolean {
  const visited = new Set<string>();
  const queue = [startId];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    if (cur === goalId) return true;
    if (visited.has(cur)) continue;
    visited.add(cur);
    for (const e of edges) {
      if (e.source === cur && !visited.has(e.target)) {
        queue.push(e.target);
      }
    }
  }
  return false;
}

let nodeSeq = 0;
function nextNodeId(kind: KindName): string {
  return `${kind}_${(nodeSeq++).toString(36)}`;
}
/// After a load, advance the id counter past the loaded set to avoid collisions.
/// Ids are `${kind}_${seq.toString(36)}`; decode the base-36 suffix so we set
/// nodeSeq above every existing id rather than just adding a length offset
/// (the offset approach re-uses ids when loaded nodes outnumber prior mints).
function bumpNodeSeq(nodes: CNode[]): void {
  let maxSeq = -1;
  for (const n of nodes) {
    const idx = n.id.lastIndexOf("_");
    if (idx >= 0) {
      const v = parseInt(n.id.slice(idx + 1), 36);
      if (Number.isFinite(v) && v > maxSeq) maxSeq = v;
    }
  }
  if (maxSeq >= nodeSeq) nodeSeq = maxSeq + 1;
}

/// Build a short red-banner summary from validation diagnostics.
function buildBanner(diag: Diagnostics): string {
  const parts: string[] = [];
  if (diag.cycle.length) parts.push(`cycle: ${diag.cycle.join(" → ")}`);
  for (const e of diag.nodeErrors) parts.push(`${e.nodeId}: ${e.message}`);
  for (const e of diag.edgeErrors) parts.push(`edge #${e.edgeIndex}: ${e.message}`);
  const head = parts.slice(0, 4).join("  •  ");
  return parts.length > 4 ? `${head}  (+${parts.length - 4} more)` : head;
}

/// Mark nodes/edges with their diagnostic status + colors; returns the banner too.
function decorateWithDiagnostics(
  nodes: CNode[],
  edges: CEdge[],
  diag: Diagnostics,
): { nodes: CNode[]; edges: CEdge[]; banner: string | null } {
  const nodeErr = new Map<string, string>();
  for (const e of diag.nodeErrors) if (!nodeErr.has(e.nodeId)) nodeErr.set(e.nodeId, e.message);
  const inCycle = new Set(diag.cycle);
  const newNodes = nodes.map((n) => {
    const msg = nodeErr.get(n.id);
    const bad = msg !== undefined || inCycle.has(n.id);
    return {
      ...n,
      data: {
        ...n.data,
        status: (bad ? "error" : "idle") as NodeStatus,
        error: msg ?? (inCycle.has(n.id) ? "participates in a cycle" : undefined),
      },
    };
  });
  const badEdge = new Set(diag.edgeErrors.map((e) => e.edgeIndex));
  const newEdges = edges.map((e, i) => {
    const base = (e.data as { color?: string } | undefined)?.color ?? "#6c6c7a";
    return { ...e, style: { ...(e.style ?? {}), stroke: badEdge.has(i) ? "#ff5a5a" : base } };
  });
  return { nodes: newNodes, edges: newEdges, banner: diagnosticsOk(diag) ? null : buildBanner(diag) };
}

/// Apply a `graph_run` rejection (serialized GraphError or a string) to the store.
function handleGraphError(
  e: unknown,
  get: () => StudioState,
  set: (p: Partial<StudioState>) => void,
): void {
  // Tauri delivers graph_run Err as a plain JSON string; parse it first so the
  // validation / node branches below can actually fire.
  let err: { kind?: string; nodeId?: string; message?: string; diagnostics?: Diagnostics } | undefined;
  if (typeof e === "string") {
    try {
      err = JSON.parse(e) as typeof err;
    } catch {
      err = { message: e };
    }
  } else {
    err = e as typeof err;
  }
  if (err && err.kind === "validation" && err.diagnostics) {
    const dec = decorateWithDiagnostics(get().graphNodes, get().graphEdges, err.diagnostics);
    set({ graphNodes: dec.nodes, graphEdges: dec.edges, graphBanner: dec.banner ?? "validation failed" });
    return;
  }
  if (err && err.kind === "node" && err.nodeId) {
    const nid = err.nodeId;
    const msg = err.message ?? "node failed";
    set({
      graphNodes: get().graphNodes.map((n) => ({
        ...n,
        data: {
          ...n.data,
          status: (n.id === nid ? "error" : "idle") as NodeStatus,
          error: n.id === nid ? msg : undefined,
        },
      })),
      graphBanner: `node ${nid}: ${msg}`,
    });
    return;
  }
  set({
    graphNodes: get().graphNodes.map((n) => ({
      ...n,
      data: { ...n.data, status: "idle" as NodeStatus },
    })),
    graphBanner: String((err && err.message) || e),
  });
}

// playback analysis throttle: the last baked frame index (and clip) we analyzed,
// so _applyTrajAt re-runs analyze only when the frame/clip changes, not every tick.
let lastAnalyzedK = -1;
let lastAnalyzedClip: TrajectoryDto | null = null;

// rAF-coalesced refresh so a slider drag fires at most one FK invoke per frame.
let pending = false;
function scheduleRefresh(get: () => StudioState) {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    void get().refreshFrames();
  });
}

// ---- recent-URDF persistence (localStorage) ----
const RECENT_KEY = "caliper.recentUrdfs";
const RECENT_CAP = 8;

/// Pure recents update: `path` moves to the front, de-duped, capped at `cap`.
/// Most-recent-first. Exported for unit testing.
export function mergeRecent(list: string[], path: string, cap = RECENT_CAP): string[] {
  return [path, ...list.filter((p) => p !== path)].slice(0, cap);
}

function persistRecents(list: string[]): void {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(list));
  } catch {
    // storage unavailable (private mode / quota) — recents stay in-memory only.
  }
}

/// Drop recents whose file no longer exists, via the backend `path_exists` probe.
/// On a probe error (e.g. command absent) the entry is KEPT, so a transient IPC
/// failure never wipes the list.
async function pruneMissingRecents(
  get: () => StudioState,
  set: (p: Partial<StudioState>) => void,
): Promise<void> {
  const list = get().recentUrdfs;
  if (list.length === 0) return;
  const alive = await Promise.all(
    list.map((p) => invoke<boolean>("path_exists", { path: p }).catch(() => true)),
  );
  const kept = list.filter((_, i) => alive[i]);
  if (kept.length !== list.length) {
    set({ recentUrdfs: kept });
    persistRecents(kept);
  }
}

// ---- test-only exports (no runtime behaviour change) ----
// These expose pure internal helpers so unit tests can drive them directly
// without going through the full Zustand store or the Tauri IPC boundary.
export { handleGraphError, bumpNodeSeq, canReach };
/** Reset the module-level node-id sequence counter to 0 (test helper only). */
export function _resetNodeSeq(): void {
  nodeSeq = 0;
}

// ---- trajectory playback clock (performance.now-driven; baked frames only) ----
let rafId = 0;
let t0 = 0;
let base = 0;
function stopClock() {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = 0;
  }
}
function tick(get: () => StudioState, set: (p: Partial<StudioState>) => void) {
  const traj = activeClip(get());
  const playing = get().playing;
  if (!traj || !playing) {
    rafId = 0;
    return;
  }
  const t = base + (performance.now() - t0) / 1000;
  if (t >= traj.duration) {
    set({ playhead: traj.duration, playing: false });
    get()._applyTrajAt(traj.duration);
    rafId = 0;
    return;
  }
  set({ playhead: t });
  get()._applyTrajAt(t);
  rafId = requestAnimationFrame(() => tick(get, set));
}
