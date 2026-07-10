// Headless unit tests for the pure contact-sim helpers (src/sim/props.ts):
// playback frame indexing for prop tracks, the default-prop builder (unique
// names, above-ground staggered spawns), the MAX_PROPS cap, and the engine-
// availability gate the whole contact UI hangs off.

import { describe, it, expect } from "vitest";
import { frameIndexAt, defaultProp, withProp, hasContactEngine, MAX_PROPS } from "./props";
import type { SimProp, PropKind } from "./props";

describe("frameIndexAt", () => {
  it("maps t=0 to the first baked frame", () => {
    expect(frameIndexAt(0, 0.02, 100)).toBe(0);
  });

  it("rounds to the NEAREST baked frame (same rule as _applyTrajAt)", () => {
    expect(frameIndexAt(0.031, 0.02, 100)).toBe(2); // 1.55 → 2
    expect(frameIndexAt(0.029, 0.02, 100)).toBe(1); // 1.45 → 1
    expect(frameIndexAt(0.02, 0.02, 100)).toBe(1);
  });

  it("clamps past the end of the track and below zero", () => {
    expect(frameIndexAt(99, 0.02, 5)).toBe(4);
    expect(frameIndexAt(-1, 0.02, 5)).toBe(0);
  });

  it("pins degenerate inputs (dt<=0, n<=0, NaN dt) to 0", () => {
    expect(frameIndexAt(1, 0, 5)).toBe(0);
    expect(frameIndexAt(1, -0.02, 5)).toBe(0);
    expect(frameIndexAt(1, NaN, 5)).toBe(0);
    expect(frameIndexAt(1, 0.02, 0)).toBe(0);
  });
});

describe("defaultProp", () => {
  it("populates exactly the size fields of its kind", () => {
    const box = defaultProp("box", []);
    expect(box.halfExtents).not.toBeNull();
    expect(box.radius).toBeNull();
    expect(box.length).toBeNull();
    const sphere = defaultProp("sphere", []);
    expect(sphere.radius).not.toBeNull();
    expect(sphere.halfExtents).toBeNull();
    expect(sphere.length).toBeNull();
    const cyl = defaultProp("cylinder", []);
    expect(cyl.radius).not.toBeNull();
    expect(cyl.length).not.toBeNull();
    expect(cyl.halfExtents).toBeNull();
  });

  it("spawns dropped ABOVE the ground plane with a light default mass", () => {
    let list: SimProp[] = [];
    for (const kind of ["box", "sphere", "cylinder", "box", "sphere"] as PropKind[]) {
      const p = defaultProp(kind, list);
      expect(p.pos[2]).toBeGreaterThan(0); // z-up world, ground at z=0
      expect(p.mass).toBeCloseTo(0.1);
      expect(p.quat).toBeNull(); // identity orientation
      list = [...list, p];
    }
  });

  it("mints names unique within the existing list, per kind", () => {
    let list: SimProp[] = [];
    for (const kind of ["box", "box", "sphere", "box"] as PropKind[]) {
      list = [...list, defaultProp(kind, list)];
    }
    expect(list.map((p) => p.name)).toEqual(["box1", "box2", "sphere1", "box3"]);
  });

  it("staggers consecutive spawn positions so props never coincide", () => {
    let list: SimProp[] = [];
    for (let i = 0; i < MAX_PROPS; i++) list = withProp(list, "sphere");
    const seen = new Set(list.map((p) => p.pos.join(",")));
    expect(seen.size).toBe(MAX_PROPS);
  });
});

describe("withProp", () => {
  it("appends without mutating the input list", () => {
    const list: SimProp[] = [];
    const next = withProp(list, "box");
    expect(next).toHaveLength(1);
    expect(list).toHaveLength(0);
  });

  it("refuses past MAX_PROPS by returning the SAME list", () => {
    let list: SimProp[] = [];
    for (let i = 0; i < MAX_PROPS; i++) list = withProp(list, "box");
    expect(list).toHaveLength(MAX_PROPS);
    const capped = withProp(list, "sphere");
    expect(capped).toBe(list); // reference-equal → callers detect the refusal
  });
});

describe("hasContactEngine", () => {
  it("is true only when the mujoco engine is listed", () => {
    expect(hasContactEngine(["builtin"])).toBe(false); // release build baseline
    expect(hasContactEngine(["builtin", "mujoco"])).toBe(true);
    expect(hasContactEngine([])).toBe(false);
  });
});
