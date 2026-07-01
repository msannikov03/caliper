// Headless unit tests for the FE ↔ Rust GraphDoc contract seam.
//
// serializeGraph() produces the JSON consumed by `graph_run` / `graph_validate`.
// Every assertion here has a direct counterpart in crates/caliper-graph/src/ir.rs:
//   - NodeKind serde tag = camelCase `"type"` field (e.g. "startConfig")
//   - GoalPose `m` = 16-element COLUMN-MAJOR SE3 homogeneous matrix
//   - Edge fromPort / toPort = NAMED string handles matching in_port_names()
//   - planRrt seed is u64 (no negative)
//
// No Tauri, no DOM, no rendering — pure TypeScript logic.

import { describe, it, expect } from "vitest";
import {
  serializeGraph,
  parseGraph,
  poseFromXyzRpy,
  xyzRpyFromPose,
  kindParams,
} from "./serialize";
import { KIND_ORDER, NODE_SPECS, defaultParams } from "./spec";
import type { KindName } from "./spec";
import type { CNode, CEdge } from "./types";

// ---- helpers ----

function makeNode(kind: KindName, id: string, ndof = 2): CNode {
  return {
    id,
    type: kind,
    position: { x: 0, y: 0 },
    data: { kind, params: defaultParams(kind, ndof), status: "idle" },
  };
}

function parseDoc(json: string): {
  nodes: Array<{ id: string; kind: Record<string, unknown> }>;
  edges: Array<Record<string, unknown>>;
} {
  return JSON.parse(json) as ReturnType<typeof parseDoc>;
}

// ---- Rust contract: node kind.type tags ----

describe("serializeGraph — kind.type tags match Rust NodeKind discriminants", () => {
  for (const kind of KIND_ORDER) {
    it(`emits type="${kind}" for ${kind} nodes`, () => {
      const node = makeNode(kind, `${kind}_0`);
      const doc = parseDoc(serializeGraph([node], [], "test"));
      expect(doc.nodes[0].kind.type).toBe(kind);
    });
  }
});

// ---- Rust contract: per-kind params ----

describe("kindParams — params match Rust variant fields", () => {
  it("startConfig emits q array", () => {
    const p = kindParams("startConfig", { q: [1, 2] });
    expect(p).toEqual({ q: [1, 2] });
  });

  it("goalPose emits m with 16 elements (column-major SE3)", () => {
    const p = kindParams("goalPose", { x: 0.1, y: 0.2, z: 0.3, roll: 0, pitch: 0, yaw: 0 });
    expect(Array.isArray(p.m)).toBe(true);
    expect((p.m as number[]).length).toBe(16);
    // column-major: m[12]=x, m[13]=y, m[14]=z, m[15]=1
    const m = p.m as number[];
    expect(m[12]).toBeCloseTo(0.1, 10);
    expect(m[13]).toBeCloseTo(0.2, 10);
    expect(m[14]).toBeCloseTo(0.3, 10);
    expect(m[15]).toBe(1);
  });

  it("namedConfig emits q and name", () => {
    const p = kindParams("namedConfig", { q: [0, 0], name: "home" });
    expect(p.name).toBe("home");
    expect(p.q).toEqual([0, 0]);
  });

  it("ik emits frame (null for empty string) and seed (null when absent)", () => {
    const p = kindParams("ik", { frame: "", seed: null });
    expect(p.frame).toBeNull();
    expect(p.seed).toBeNull();
  });

  it("ik emits frame string when non-empty", () => {
    const p = kindParams("ik", { frame: "tool0", seed: [0, 0] });
    expect(p.frame).toBe("tool0");
    expect(p.seed).toEqual([0, 0]);
  });

  it("moveJ emits empty params (no required fields)", () => {
    const p = kindParams("moveJ", {});
    expect(Object.keys(p)).toHaveLength(0);
  });

  it("moveL emits frame (nullable)", () => {
    const p = kindParams("moveL", { frame: "" });
    expect(p.frame).toBeNull();
    const p2 = kindParams("moveL", { frame: "tip" });
    expect(p2.frame).toBe("tip");
  });

  it("planRrt emits seed, ground (nullable), boxes", () => {
    const p = kindParams("planRrt", { seed: 42, groundOn: true, ground: -0.1, boxes: [] });
    expect(p.seed).toBe(42);
    expect(p.ground).toBeCloseTo(-0.1, 10);
    expect(p.boxes).toEqual([]);
  });

  it("planRrt: groundOn=false emits ground=null", () => {
    const p = kindParams("planRrt", { seed: 1, groundOn: false, ground: 0, boxes: [] });
    expect(p.ground).toBeNull();
  });

  it("planRrt seed is clamped to >=0 (matches Rust u64)", () => {
    const p = kindParams("planRrt", { seed: -7, groundOn: false, ground: 0, boxes: [] });
    expect((p.seed as number) >= 0).toBe(true);
  });

  it("control emits kp and kd", () => {
    const p = kindParams("control", { kp: 200, kd: 40 });
    expect(p.kp).toBe(200);
    expect(p.kd).toBe(40);
  });

  it("gravityDrop: gravityOn=true emits gravity=null (engine default = earth)", () => {
    const p = kindParams("gravityDrop", { gravityOn: true, duration: 2, dt: 0.001 });
    expect(p.gravity).toBeNull();
    expect(p.duration).toBe(2);
    expect(p.dt).toBe(0.001);
  });

  it("gravityDrop: gravityOn=false emits gravity=[0,0,0] (no gravity)", () => {
    const p = kindParams("gravityDrop", { gravityOn: false, duration: 2, dt: 0.001 });
    expect(p.gravity).toEqual([0, 0, 0]);
  });

  it("collisionCheck emits ground (nullable) and boxes", () => {
    const p = kindParams("collisionCheck", { groundOn: false, ground: 0, boxes: [] });
    expect(p.ground).toBeNull();
    const p2 = kindParams("collisionCheck", { groundOn: true, ground: -0.05, boxes: [] });
    expect(p2.ground).toBeCloseTo(-0.05, 10);
  });

  it("view emits empty params", () => {
    const p = kindParams("view", {});
    expect(Object.keys(p)).toHaveLength(0);
  });

  it("scope emits signal string", () => {
    const p = kindParams("scope", { signal: "q2" });
    expect(p.signal).toBe("q2");
  });
});

// ---- Rust contract: edge port names ----

describe("serializeGraph — edge fromPort/toPort are named strings (not indices)", () => {
  it("startConfig.config → moveJ.start edge uses named port handles", () => {
    const sc = makeNode("startConfig", "sc_0");
    const mj = makeNode("moveJ", "mj_0");
    const edge: CEdge = {
      id: "e0",
      source: "sc_0",
      target: "mj_0",
      sourceHandle: "config",
      targetHandle: "start",
    };
    const doc = parseDoc(serializeGraph([sc, mj], [edge], "test"));
    const e = doc.edges[0];
    expect(e.fromPort).toBe("config");
    expect(e.toPort).toBe("start");
    expect(e.from).toBe("sc_0");
    expect(e.to).toBe("mj_0");
  });

  it("goalPose.pose → ik.pose edge uses named port handles", () => {
    const gp = makeNode("goalPose", "gp_0");
    const ik = makeNode("ik", "ik_0");
    const edge: CEdge = {
      id: "e1",
      source: "gp_0",
      target: "ik_0",
      sourceHandle: "pose",
      targetHandle: "pose",
    };
    const doc = parseDoc(serializeGraph([gp, ik], [edge], "test"));
    const e = doc.edges[0];
    expect(e.fromPort).toBe("pose");
    expect(e.toPort).toBe("pose");
  });

  it("view.clip → view uses named 'clip' handle", () => {
    const mj = makeNode("moveJ", "mj_0");
    const vw = makeNode("view", "vw_0");
    const edge: CEdge = {
      id: "e2",
      source: "mj_0",
      target: "vw_0",
      sourceHandle: "clip",
      targetHandle: "clip",
    };
    const doc = parseDoc(serializeGraph([mj, vw], [edge], "test"));
    expect(doc.edges[0].fromPort).toBe("clip");
    expect(doc.edges[0].toPort).toBe("clip");
  });
});

// ---- all known Rust port names are present in NODE_SPECS ----

describe("NODE_SPECS port names match Rust in_port_names / out_port_names", () => {
  // Cross-check the catalogue against the expected Rust port names.
  const RUST_IN_PORTS: Partial<Record<KindName, string[]>> = {
    ik: ["pose", "seed"],
    moveJ: ["start", "goal"],
    moveL: ["start", "goal"],
    planRrt: ["start", "goal"],
    control: ["start", "goal"],
    gravityDrop: ["start"],
    collisionCheck: ["config"],
    view: ["clip"],
    scope: ["clip"],
  };
  const RUST_OUT_PORTS: Partial<Record<KindName, string[]>> = {
    startConfig: ["config"],
    goalPose: ["pose"],
    namedConfig: ["config"],
    ik: ["config"],
    moveJ: ["clip"],
    moveL: ["clip"],
    planRrt: ["clip"],
    control: ["clip"],
    gravityDrop: ["clip"],
    collisionCheck: ["report"],
  };

  for (const [kind, names] of Object.entries(RUST_IN_PORTS)) {
    it(`${kind} input ports: ${names.join(", ")}`, () => {
      const spec = NODE_SPECS[kind as KindName];
      expect(spec.inputs.map((p) => p.name)).toEqual(names);
    });
  }
  for (const [kind, names] of Object.entries(RUST_OUT_PORTS)) {
    it(`${kind} output ports: ${names.join(", ")}`, () => {
      const spec = NODE_SPECS[kind as KindName];
      expect(spec.outputs.map((o) => o.name)).toEqual(names);
    });
  }
});

// ---- round-trip: parseGraph(serializeGraph(x)) ----

describe("parseGraph round-trip", () => {
  it("nodes, params, and layout survive serialize → parse", () => {
    const nodes: CNode[] = [
      { ...makeNode("startConfig", "sc_0"), position: { x: 100, y: 200 } },
      { ...makeNode("ik", "ik_0"), position: { x: 300, y: 100 } },
      { ...makeNode("scope", "sc2_0"), position: { x: 500, y: 300 } },
    ];
    // patch ik params with a real frame name
    nodes[1] = {
      ...nodes[1],
      data: { ...nodes[1].data, params: { frame: "tool0", seed: null } },
    };
    const edges: CEdge[] = [
      {
        id: "e0",
        source: "sc_0",
        target: "ik_0",
        sourceHandle: "config",
        targetHandle: "seed",
      },
    ];
    const json = serializeGraph(nodes, edges, "my-graph", "panda");
    const back = parseGraph(json, 2);

    // nodes round-trip
    expect(back.nodes).toHaveLength(3);
    expect(back.nodes.map((n) => n.id)).toEqual(["sc_0", "ik_0", "sc2_0"]);
    expect(back.nodes.map((n) => n.data.kind)).toEqual(["startConfig", "ik", "scope"]);
    expect(back.nodes[0].position).toEqual({ x: 100, y: 200 });
    expect(back.nodes[1].position).toEqual({ x: 300, y: 100 });

    // ik frame param round-trips
    expect(back.nodes[1].data.params.frame).toBe("tool0");

    // edge round-trip
    expect(back.edges).toHaveLength(1);
    expect(back.edges[0].source).toBe("sc_0");
    expect(back.edges[0].target).toBe("ik_0");
    expect(back.edges[0].sourceHandle).toBe("config");
    expect(back.edges[0].targetHandle).toBe("seed");

    // graph name
    expect(back.name).toBe("my-graph");
  });

  it("unknown node kinds in stored JSON are silently dropped", () => {
    const json = JSON.stringify({
      nodes: [
        { id: "n0", kind: { type: "UNKNOWN_KIND" } },
        { id: "n1", kind: { type: "startConfig", q: [0, 0] } },
      ],
      edges: [],
      metadata: {},
    });
    const back = parseGraph(json, 2);
    expect(back.nodes).toHaveLength(1);
    expect(back.nodes[0].id).toBe("n1");
  });

  it("edge referencing a missing node is dropped", () => {
    const json = JSON.stringify({
      nodes: [{ id: "sc", kind: { type: "startConfig", q: [0] } }],
      edges: [{ from: "sc", fromPort: "config", to: "MISSING", toPort: "start" }],
      metadata: {},
    });
    const back = parseGraph(json, 1);
    expect(back.edges).toHaveLength(0);
  });
});

// ---- goalPose column-major SE3 round-trip ----

describe("poseFromXyzRpy / xyzRpyFromPose — column-major SE3 round-trip", () => {
  it("translation lands in m[12..14] (column-major homogeneous convention)", () => {
    const m = poseFromXyzRpy(0.1, 0.2, 0.3, 0, 0, 0);
    expect(m).toHaveLength(16);
    expect(m[12]).toBeCloseTo(0.1, 12);
    expect(m[13]).toBeCloseTo(0.2, 12);
    expect(m[14]).toBeCloseTo(0.3, 12);
    expect(m[15]).toBe(1);
    // identity rotation: m[0]=m[5]=m[10]=1, off-diagonal rotation entries=0
    expect(m[0]).toBeCloseTo(1, 12);
    expect(m[5]).toBeCloseTo(1, 12);
    expect(m[10]).toBeCloseTo(1, 12);
    expect(m[1]).toBeCloseTo(0, 12);
    expect(m[2]).toBeCloseTo(0, 12);
  });

  it("round-trips xyz+rpy exactly (within floating-point tolerance)", () => {
    const cases: [number, number, number, number, number, number][] = [
      [0.1, 0.2, 0.3, 0, 0, 0],
      [0, 0, 0, 1.0, 0.5, -0.3],
      [-0.5, 0.7, 0.1, 0.2, 0.4, 0.6],
      [1.0, -1.0, 0.5, 0, Math.PI / 4, 0],
    ];
    for (const [x, y, z, roll, pitch, yaw] of cases) {
      const m = poseFromXyzRpy(x, y, z, roll, pitch, yaw);
      const back = xyzRpyFromPose(m);
      expect(back.x).toBeCloseTo(x, 10);
      expect(back.y).toBeCloseTo(y, 10);
      expect(back.z).toBeCloseTo(z, 10);
      expect(back.roll).toBeCloseTo(roll, 10);
      expect(back.pitch).toBeCloseTo(pitch, 10);
      expect(back.yaw).toBeCloseTo(yaw, 10);
    }
  });

  it("produced matrix is a valid rotation (columns are unit and orthogonal)", () => {
    const m = poseFromXyzRpy(0, 0, 0, 0.3, 0.5, -0.7);
    // Extract the 3 rotation columns (col-major: indices 0-2, 4-6, 8-10)
    const c0 = [m[0], m[1], m[2]];
    const c1 = [m[4], m[5], m[6]];
    const c2 = [m[8], m[9], m[10]];
    const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
    expect(norm(c0)).toBeCloseTo(1, 12);
    expect(norm(c1)).toBeCloseTo(1, 12);
    expect(norm(c2)).toBeCloseTo(1, 12);
    expect(dot(c0, c1)).toBeCloseTo(0, 12);
    expect(dot(c0, c2)).toBeCloseTo(0, 12);
    expect(dot(c1, c2)).toBeCloseTo(0, 12);
  });

  it("goalPose serialization embeds the column-major matrix in kind.m", () => {
    const node = makeNode("goalPose", "gp_0");
    node.data.params = { x: 0.5, y: 0, z: 0.4, roll: 0, pitch: 0, yaw: 0 };
    const doc = parseDoc(serializeGraph([node], [], "t"));
    const m = doc.nodes[0].kind.m as number[];
    expect(m).toHaveLength(16);
    expect(m[12]).toBeCloseTo(0.5, 10);
    expect(m[14]).toBeCloseTo(0.4, 10);
    expect(m[15]).toBe(1);
  });
});
