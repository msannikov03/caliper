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
}

/** column-major 4x4 (THREE.Matrix4 element order), exactly what fromArray wants. */
export type Mat4 = number[]; // length 16

export interface IkSolution {
  success: boolean;
  q: number[];
  residual: number;
}

interface StudioState {
  // robot + configuration
  robot: RobotInfo | null;
  q: number[]; // length ndof — the single source of truth for configuration
  frames: Mat4[]; // length frames.length — world poses, recomputed by the engine

  // status
  loading: boolean;
  error: string | null;
  ikOk: boolean | null;
  ikResidual: number | null;

  // internal: monotonic request id so older FK replies can't clobber newer ones
  _reqId: number;

  // actions
  loadRobot: (path: string) => Promise<void>;
  setJoint: (i: number, v: number) => void;
  refreshFrames: () => Promise<void>;
  solveIk: (targetColMajor: Mat4) => Promise<void>;
}

export const useStore = create<StudioState>((set, get) => ({
  robot: null,
  q: [],
  frames: [],
  loading: false,
  error: null,
  ikOk: null,
  ikResidual: null,
  _reqId: 0,

  async loadRobot(path) {
    set({ loading: true, error: null });
    try {
      const robot = await invoke<RobotInfo>("robot_info", { path });
      set({
        robot,
        q: new Array(robot.ndof).fill(0),
        frames: [],
        ikOk: null,
        ikResidual: null,
      });
      await get().refreshFrames();
    } catch (e) {
      set({ error: String(e), robot: null });
    } finally {
      set({ loading: false });
    }
  },

  setJoint(i, v) {
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
    } catch (e) {
      set({ error: String(e) });
    }
  },

  async solveIk(targetColMajor) {
    const { q, robot } = get();
    if (!robot) return;
    const frameName = robot.frames[robot.tip].name;
    // share the monotonic guard with refreshFrames so an out-of-order IK reply
    // from a fast drag can't write a stale q (or trigger a stale FK refresh).
    const reqId = get()._reqId + 1;
    set({ _reqId: reqId });
    try {
      const res = await invoke<IkSolution>("solve_ik", {
        req: { target: targetColMajor, seed: q, frame: frameName },
      });
      if (get()._reqId !== reqId) return; // a newer request superseded us
      // arm follows even on best-effort failure, but flag it for the HUD.
      set({ q: res.q, ikOk: res.success, ikResidual: res.residual, error: null });
      await get().refreshFrames();
    } catch (e) {
      set({ error: String(e) });
    }
  },
}));

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
