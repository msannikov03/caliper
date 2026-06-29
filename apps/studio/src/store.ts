import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";

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

export interface RobotInfo {
  name: string;
  ndof: number;
  jointNames: string[];
  jointKinds: JointKind[]; // parallel to jointNames
  limits: ([number, number] | null)[]; // parallel to jointNames
  frames: FrameInfo[];
  tip: number; // index into frames
  hasInertia: boolean; // dynamics available (every link has <inertial>)
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
export type StudioMode = "jog" | "motion" | "simulate";
export interface NamedPoseDto {
  name: string;
  q: number[];
}

interface StudioState {
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
      });
      await get().refreshFrames();
      await get().refreshPoses();
    } catch (e) {
      set({ error: String(e), robot: null });
    } finally {
      set({ loading: false });
    }
  },

  setJoint(i, v) {
    if (get().playing || get().mode === "simulate") return; // sim/playback own the pose
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
    if (!robot || get().playing || get().mode === "simulate") return; // gizmo inert
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
}));

/// The active playback clip: a baked sim rollout takes precedence over a motion traj.
function activeClip(s: StudioState): TrajectoryDto | null {
  return s.simTraj ?? s.traj;
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
