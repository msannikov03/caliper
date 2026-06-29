// The node/port catalogue for Graph mode — the single source of truth shared by
// the custom node components, the connection-type guard, and (de)serialization.
// Mirrors caliper-graph::ir::NodeKind::{in_ports,out_ports} EXACTLY.

/** Wire types (caliper-graph::ir::PortType). UI handle colors keyed off these. */
export type PortType = "config" | "pose" | "clip" | "report";

/** The 12 node kinds (caliper-graph::ir::NodeKind `"type"` tags). */
export type KindName =
  | "startConfig"
  | "goalPose"
  | "namedConfig"
  | "ik"
  | "moveJ"
  | "moveL"
  | "planRrt"
  | "control"
  | "gravityDrop"
  | "collisionCheck"
  | "view"
  | "scope";

export type NodeCategory = "source" | "compute" | "sink";

export interface InPortDef {
  name: string;
  /** Accepted wire types (a union, e.g. planRrt.goal accepts config|pose). */
  types: PortType[];
  required: boolean;
}
export interface OutPortDef {
  name: string;
  type: PortType;
}
export interface NodeSpec {
  kind: KindName;
  label: string;
  category: NodeCategory;
  inputs: InPortDef[];
  outputs: OutPortDef[];
  /** A one-line description shown in the palette + node body. */
  blurb: string;
}

const cfg: PortType[] = ["config"];
const pose: PortType[] = ["pose"];
const clip: PortType[] = ["clip"];
const cfgOrPose: PortType[] = ["config", "pose"];

export const NODE_SPECS: Record<KindName, NodeSpec> = {
  startConfig: {
    kind: "startConfig",
    label: "Start Config",
    category: "source",
    inputs: [],
    outputs: [{ name: "config", type: "config" }],
    blurb: "fixed joint configuration",
  },
  goalPose: {
    kind: "goalPose",
    label: "Goal Pose",
    category: "source",
    inputs: [],
    outputs: [{ name: "pose", type: "pose" }],
    blurb: "Cartesian target (xyz + rpy)",
  },
  namedConfig: {
    kind: "namedConfig",
    label: "Named Config",
    category: "source",
    inputs: [],
    outputs: [{ name: "config", type: "config" }],
    blurb: "a labelled configuration",
  },
  ik: {
    kind: "ik",
    label: "IK",
    category: "compute",
    inputs: [
      { name: "pose", types: pose, required: true },
      { name: "seed", types: cfg, required: false },
    ],
    outputs: [{ name: "config", type: "config" }],
    blurb: "inverse kinematics → config",
  },
  moveJ: {
    kind: "moveJ",
    label: "Move J",
    category: "compute",
    inputs: [
      { name: "start", types: cfg, required: true },
      { name: "goal", types: cfg, required: true },
    ],
    outputs: [{ name: "clip", type: "clip" }],
    blurb: "jerk-limited joint move",
  },
  moveL: {
    kind: "moveL",
    label: "Move L",
    category: "compute",
    inputs: [
      { name: "start", types: cfg, required: true },
      { name: "goal", types: pose, required: true },
    ],
    outputs: [{ name: "clip", type: "clip" }],
    blurb: "straight-line Cartesian move",
  },
  planRrt: {
    kind: "planRrt",
    label: "Plan RRT",
    category: "compute",
    inputs: [
      { name: "start", types: cfg, required: true },
      { name: "goal", types: cfgOrPose, required: true },
    ],
    outputs: [{ name: "clip", type: "clip" }],
    blurb: "collision-aware RRT-Connect",
  },
  control: {
    kind: "control",
    label: "Control",
    category: "compute",
    inputs: [
      { name: "start", types: cfg, required: true },
      { name: "goal", types: cfg, required: true },
    ],
    outputs: [{ name: "clip", type: "clip" }],
    blurb: "computed-torque rollout (needs inertia)",
  },
  gravityDrop: {
    kind: "gravityDrop",
    label: "Gravity Drop",
    category: "compute",
    inputs: [{ name: "start", types: cfg, required: true }],
    outputs: [{ name: "clip", type: "clip" }],
    blurb: "passive dynamics drop (needs inertia)",
  },
  collisionCheck: {
    kind: "collisionCheck",
    label: "Collision Check",
    category: "compute",
    inputs: [{ name: "config", types: cfg, required: true }],
    outputs: [{ name: "report", type: "report" }],
    blurb: "self / world collision query",
  },
  view: {
    kind: "view",
    label: "View",
    category: "sink",
    inputs: [{ name: "clip", types: clip, required: true }],
    outputs: [],
    blurb: "drives the 3D preview",
  },
  scope: {
    kind: "scope",
    label: "Scope",
    category: "sink",
    inputs: [{ name: "clip", types: clip, required: true }],
    outputs: [],
    blurb: "plot a signal vs time",
  },
};

export const KIND_ORDER: KindName[] = [
  "startConfig",
  "goalPose",
  "namedConfig",
  "ik",
  "moveJ",
  "moveL",
  "planRrt",
  "control",
  "gravityDrop",
  "collisionCheck",
  "view",
  "scope",
];

// ---- handle colors (config=amber, pose=blue, clip=green, report=grey) ----
export const PORT_COLORS: Record<PortType, string> = {
  config: "#f5a623",
  pose: "#5a9bff",
  clip: "#5aff7a",
  report: "#9a9aa6",
};
/** A union input port (e.g. config|pose) gets its own violet handle. */
export const UNION_COLOR = "#b39ddb";

export function inHandleColor(p: InPortDef): string {
  return p.types.length > 1 ? UNION_COLOR : PORT_COLORS[p.types[0]];
}

/** The single wire type emitted by a named output port (or undefined). */
export function outPortType(kind: KindName, port: string): PortType | undefined {
  return NODE_SPECS[kind].outputs.find((o) => o.name === port)?.type;
}
/** The accepted wire types of a named input port (or undefined). */
export function inPortTypes(kind: KindName, port: string): PortType[] | undefined {
  return NODE_SPECS[kind].inputs.find((i) => i.name === port)?.types;
}

/** Scope signal options for a given dof count. */
export function signalOptions(ndof: number): string[] {
  const base = ["t", "tip_x", "tip_y", "tip_z", "energy"];
  const q: string[] = [];
  for (let i = 0; i < ndof; i++) q.push(`q${i}`);
  const qd: string[] = [];
  for (let i = 0; i < ndof; i++) qd.push(`qd${i}`);
  return [...q, ...qd, ...base];
}

/** A fresh UI parameter bag for a newly-added node of `kind`. */
export function defaultParams(kind: KindName, ndof: number): Record<string, unknown> {
  const zeros = new Array(Math.max(ndof, 0)).fill(0);
  switch (kind) {
    case "startConfig":
      return { q: zeros };
    case "namedConfig":
      return { q: zeros, name: "home" };
    case "goalPose":
      return { x: 0.3, y: 0, z: 0.3, roll: 0, pitch: 0, yaw: 0 };
    case "ik":
      return { frame: "" };
    case "moveJ":
      return {};
    case "moveL":
      return { frame: "" };
    case "planRrt":
      return { seed: 1, groundOn: false, ground: 0, boxes: [] };
    case "control":
      return { kp: 100, kd: 20 };
    case "gravityDrop":
      return { gravityOn: true, duration: 2, dt: 0.001 };
    case "collisionCheck":
      return { groundOn: false, ground: 0, boxes: [] };
    case "view":
      return {};
    case "scope":
      return { signal: "q0" };
  }
}
