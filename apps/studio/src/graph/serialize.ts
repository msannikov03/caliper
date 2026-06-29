// Map between the xyflow editor state and the persisted `.caliper-graph.json`
// GraphDoc (caliper-graph::ir). serializeGraph() produces the JSON the Tauri
// `graph_run` / `graph_validate` / `save_graph` commands consume; parseGraph()
// rebuilds the editor (positions + params) from a loaded document.

import type { CNode, CEdge } from "./types";
import type { KindName } from "./spec";
import { NODE_SPECS, KIND_ORDER, defaultParams } from "./spec";

type Box = [[number, number, number], [number, number, number]];

function num(v: unknown, d = 0): number {
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : d;
}
function strOrNull(v: unknown): string | null {
  const s = typeof v === "string" ? v.trim() : "";
  return s.length ? s : null;
}
function asNumArray(v: unknown): number[] {
  return Array.isArray(v) ? v.map((x) => num(x)) : [];
}
function asBoxes(v: unknown): Box[] {
  if (!Array.isArray(v)) return [];
  return v.map((b) => {
    const pair = b as unknown[];
    const c = asNumArray(pair?.[0]);
    const h = asNumArray(pair?.[1]);
    return [
      [c[0] ?? 0, c[1] ?? 0, c[2] ?? 0],
      [h[0] ?? 0.1, h[1] ?? 0.1, h[2] ?? 0.1],
    ] as Box;
  });
}

// ===== SE3 <-> 16-element column-major homogeneous matrix (matches ir.rs) =====
// Rotation is fixed-axis URDF rpy: R = Rz(yaw)·Ry(pitch)·Rx(roll).
export function poseFromXyzRpy(
  x: number,
  y: number,
  z: number,
  roll: number,
  pitch: number,
  yaw: number,
): number[] {
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const r00 = cy * cp, r01 = cy * sp * sr - sy * cr, r02 = cy * sp * cr + sy * sr;
  const r10 = sy * cp, r11 = sy * sp * sr + cy * cr, r12 = sy * sp * cr - cy * sr;
  const r20 = -sp, r21 = cp * sr, r22 = cp * cr;
  // column-major: [col0(3) 0, col1(3) 0, col2(3) 0, trans(3) 1]
  return [r00, r10, r20, 0, r01, r11, r21, 0, r02, r12, r22, 0, x, y, z, 1];
}
export function xyzRpyFromPose(m: number[]): {
  x: number; y: number; z: number; roll: number; pitch: number; yaw: number;
} {
  const r00 = m[0], r10 = m[1], r20 = m[2];
  const r21 = m[6], r22 = m[10];
  return {
    x: m[12] ?? 0,
    y: m[13] ?? 0,
    z: m[14] ?? 0,
    yaw: Math.atan2(r10, r00),
    pitch: Math.atan2(-r20, Math.hypot(r00, r10)),
    roll: Math.atan2(r21, r22),
  };
}

/** Project a node's UI params to the kind's JSON params (caliper-graph::ir). */
export function kindParams(kind: KindName, p: Record<string, unknown>): Record<string, unknown> {
  switch (kind) {
    case "startConfig":
      return { q: asNumArray(p.q) };
    case "namedConfig":
      return { q: asNumArray(p.q), name: typeof p.name === "string" ? p.name : "" };
    case "goalPose":
      return {
        m: poseFromXyzRpy(num(p.x), num(p.y), num(p.z), num(p.roll), num(p.pitch), num(p.yaw)),
      };
    case "ik":
      return { frame: strOrNull(p.frame), seed: Array.isArray(p.seed) ? asNumArray(p.seed) : null };
    case "moveJ":
      return {};
    case "moveL":
      return { frame: strOrNull(p.frame) };
    case "planRrt":
      return {
        seed: Math.max(0, Math.trunc(num(p.seed, 1))),
        ground: p.groundOn ? num(p.ground) : null,
        boxes: asBoxes(p.boxes),
      };
    case "control":
      return { kp: num(p.kp, 100), kd: num(p.kd, 20) };
    case "gravityDrop":
      // None ⇒ engine default (earth gravity); [0,0,0] ⇒ gravity off.
      return {
        gravity: p.gravityOn ? null : [0, 0, 0],
        duration: num(p.duration, 2),
        dt: num(p.dt, 0.001),
      };
    case "collisionCheck":
      return { ground: p.groundOn ? num(p.ground) : null, boxes: asBoxes(p.boxes) };
    case "view":
      return {};
    case "scope":
      return { signal: typeof p.signal === "string" ? p.signal : "q0" };
  }
}

/** Inverse of kindParams: rebuild UI params from a loaded kind JSON object. */
function paramsFromKind(kind: KindName, k: Record<string, unknown>): Record<string, unknown> {
  switch (kind) {
    case "startConfig":
      return { q: asNumArray(k.q) };
    case "namedConfig":
      return { q: asNumArray(k.q), name: typeof k.name === "string" ? k.name : "home" };
    case "goalPose":
      return xyzRpyFromPose(asNumArray(k.m));
    case "ik":
      return { frame: typeof k.frame === "string" ? k.frame : "", seed: k.seed ?? null };
    case "moveJ":
      return {};
    case "moveL":
      return { frame: typeof k.frame === "string" ? k.frame : "" };
    case "planRrt":
      return {
        seed: num(k.seed, 1),
        groundOn: k.ground != null,
        ground: num(k.ground),
        boxes: asBoxes(k.boxes),
      };
    case "control":
      return { kp: num(k.kp, 100), kd: num(k.kd, 20) };
    case "gravityDrop": {
      const g = k.gravity;
      const off = Array.isArray(g) && g.every((v) => num(v) === 0);
      return { gravityOn: !off, duration: num(k.duration, 2), dt: num(k.dt, 0.001) };
    }
    case "collisionCheck":
      return { groundOn: k.ground != null, ground: num(k.ground), boxes: asBoxes(k.boxes) };
    case "view":
      return {};
    case "scope":
      return { signal: typeof k.signal === "string" ? k.signal : "q0" };
  }
}

/** Build the GraphDoc JSON string for the Tauri commands. */
export function serializeGraph(
  nodes: CNode[],
  edges: CEdge[],
  name: string,
  robot?: string,
): string {
  const layout: Record<string, { x: number; y: number }> = {};
  for (const n of nodes) layout[n.id] = { x: n.position.x, y: n.position.y };
  const doc = {
    nodes: nodes.map((n) => ({
      id: n.id,
      kind: { type: n.data.kind, ...kindParams(n.data.kind, n.data.params) },
    })),
    edges: edges.map((e) => ({
      from: e.source,
      fromPort: e.sourceHandle ?? 0,
      to: e.target,
      toPort: e.targetHandle ?? 0,
    })),
    metadata: {
      ...(name ? { name } : {}),
      ...(robot ? { robot } : {}),
      layout,
    },
  };
  return JSON.stringify(doc);
}

interface RawNode {
  id: string;
  kind: { type: string } & Record<string, unknown>;
}
interface RawEdge {
  from: string;
  fromPort?: string | number;
  to: string;
  toPort?: string | number;
}

function isKind(t: string): t is KindName {
  return KIND_ORDER.includes(t as KindName);
}

/** Resolve a stored fromPort/toPort (name or index) to a port name string. */
function portName(ref: string | number | undefined, names: string[]): string | undefined {
  if (ref === undefined) return names[0];
  if (typeof ref === "number") return names[ref];
  return names.includes(ref) ? ref : names[0];
}

/** Rebuild editor nodes/edges from a loaded GraphDoc JSON string. */
export function parseGraph(
  json: string,
  ndof: number,
): { nodes: CNode[]; edges: CEdge[]; name: string } {
  const doc = JSON.parse(json) as {
    nodes?: RawNode[];
    edges?: RawEdge[];
    metadata?: { name?: string; layout?: Record<string, { x: number; y: number }> };
  };
  const layout = doc.metadata?.layout ?? {};
  const rawNodes = (doc.nodes ?? []).filter((n) => isKind(n.kind.type));
  const nodes: CNode[] = rawNodes.map((n, i) => {
    const kind = n.kind.type as KindName;
    const pos = layout[n.id] ?? { x: 60 + (i % 4) * 230, y: 60 + Math.floor(i / 4) * 200 };
    return {
      id: n.id,
      type: kind,
      position: pos,
      data: {
        kind,
        params: { ...defaultParams(kind, ndof), ...paramsFromKind(kind, n.kind) },
        status: "idle",
      },
    };
  });
  const byId = new Map(rawNodes.map((n) => [n.id, n.kind.type as KindName]));
  const edges: CEdge[] = [];
  (doc.edges ?? []).forEach((e, i) => {
    const fk = byId.get(e.from);
    const tk = byId.get(e.to);
    if (!fk || !tk) return;
    const sh = portName(e.fromPort, NODE_SPECS[fk].outputs.map((o) => o.name));
    const th = portName(e.toPort, NODE_SPECS[tk].inputs.map((p) => p.name));
    edges.push({
      id: `e${i}_${e.from}_${e.to}`,
      source: e.from,
      target: e.to,
      sourceHandle: sh,
      targetHandle: th,
    });
  });
  return { nodes, edges, name: doc.metadata?.name ?? "" };
}
