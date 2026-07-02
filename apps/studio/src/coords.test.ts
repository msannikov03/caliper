// Headless unit tests for the URDF ↔ three.js display-frame coordinate helpers.
//
// DISPLAY_UP = -90° rotation around X: maps URDF +Z → three.js display +Y.
// DISPLAY_UP_INV is its exact inverse.
//
// The IK gizmo lives inside the DISPLAY_UP group; to feed the engine a URDF-world
// target we must left-multiply the gizmo's world matrix by DISPLAY_UP_INV.
// These tests verify that round-trip is numerically invertible.
//
// three.js Matrix4 is pure JavaScript math — no WebGL, no canvas, runs in jsdom.

import { describe, it, expect } from "vitest";
import * as THREE from "three";
import { DISPLAY_UP, DISPLAY_UP_INV } from "./coords";
import { composeWorld, primitiveSpec } from "./three/Visuals";
import type { VisualInfo } from "./store";

// ---- DISPLAY_UP correctness ----

describe("DISPLAY_UP — -90° rotation around X", () => {
  it("maps URDF +Z to three.js display +Y", () => {
    const v = new THREE.Vector3(0, 0, 1); // +Z in URDF world
    v.applyMatrix4(DISPLAY_UP);
    expect(v.x).toBeCloseTo(0, 12);
    expect(v.y).toBeCloseTo(1, 12); // must land on +Y in display space
    expect(v.z).toBeCloseTo(0, 12);
  });

  it("maps URDF +Y to three.js display -Z (secondary consequence of -90° around X)", () => {
    const v = new THREE.Vector3(0, 1, 0); // +Y URDF
    v.applyMatrix4(DISPLAY_UP);
    expect(v.x).toBeCloseTo(0, 12);
    expect(v.y).toBeCloseTo(0, 12);
    expect(v.z).toBeCloseTo(-1, 12); // -Z in display space
  });

  it("leaves the +X axis unchanged (rotation is around X)", () => {
    const v = new THREE.Vector3(1, 0, 0);
    v.applyMatrix4(DISPLAY_UP);
    expect(v.x).toBeCloseTo(1, 12);
    expect(v.y).toBeCloseTo(0, 12);
    expect(v.z).toBeCloseTo(0, 12);
  });
});

// ---- DISPLAY_UP_INV is the exact inverse ----

describe("DISPLAY_UP_INV — mathematical inverse", () => {
  it("DISPLAY_UP * DISPLAY_UP_INV equals the identity matrix", () => {
    // THREE convention: A.clone().multiply(B) = A * B
    const product = DISPLAY_UP.clone().multiply(DISPLAY_UP_INV);
    const identity = new THREE.Matrix4(); // identity by default
    for (let i = 0; i < 16; i++) {
      expect(product.elements[i]).toBeCloseTo(identity.elements[i], 12);
    }
  });

  it("DISPLAY_UP_INV * DISPLAY_UP also equals the identity matrix", () => {
    const product = DISPLAY_UP_INV.clone().multiply(DISPLAY_UP);
    const identity = new THREE.Matrix4();
    for (let i = 0; i < 16; i++) {
      expect(product.elements[i]).toBeCloseTo(identity.elements[i], 12);
    }
  });
});

// ---- IK target recovery round-trip ----

describe("IK target recovery — DISPLAY_UP_INV * DISPLAY_UP * M = M", () => {
  // The gizmo lives inside the DISPLAY_UP group, so:
  //   engine tip pose T_tip (URDF world)
  //   → fed to gizmo as-is (gizmo's local matrix = T_tip)
  //   → gizmo world = DISPLAY_UP * T_tip
  //   → after drag: gizmo world = DISPLAY_UP * T_target
  //   → URDF-world target = DISPLAY_UP_INV * (DISPLAY_UP * T_target) = T_target

  function roundTrip(M: THREE.Matrix4): THREE.Matrix4 {
    const transformed = DISPLAY_UP.clone().multiply(M); // DISPLAY_UP * M
    return DISPLAY_UP_INV.clone().multiply(transformed); // DISPLAY_UP_INV * DISPLAY_UP * M
  }

  function expectMatricesClose(a: THREE.Matrix4, b: THREE.Matrix4, decimals = 10) {
    for (let i = 0; i < 16; i++) {
      expect(a.elements[i]).toBeCloseTo(b.elements[i], decimals);
    }
  }

  it("round-trips the identity matrix", () => {
    const M = new THREE.Matrix4();
    expectMatricesClose(roundTrip(M), M);
  });

  it("round-trips a pure translation", () => {
    const M = new THREE.Matrix4().makeTranslation(0.3, -0.1, 0.7);
    expectMatricesClose(roundTrip(M), M);
  });

  it("round-trips a pure rotation (30° around Z)", () => {
    const M = new THREE.Matrix4().makeRotationZ(Math.PI / 6);
    expectMatricesClose(roundTrip(M), M);
  });

  it("round-trips a realistic robot tip pose (rotation + translation)", () => {
    // A representative end-effector pose: ~45° rotation around Y + offset.
    const M = new THREE.Matrix4().set(
      0.707, 0, 0.707, 0.4, // row 0 (stored row-major in .set)
      0, 1, 0, 0.0,
      -0.707, 0, 0.707, 0.3,
      0, 0, 0, 1,
    );
    expectMatricesClose(roundTrip(M), M);
  });

  it("round-trips a matrix with non-trivial roll/pitch/yaw", () => {
    // Build via THREE's Euler to get a realistic SO(3).
    const M = new THREE.Matrix4().makeRotationFromEuler(
      new THREE.Euler(0.3, 0.5, -0.7, "XYZ"),
    );
    M.setPosition(0.1, 0.2, -0.3);
    expectMatricesClose(roundTrip(M), M);
  });
});

// ---- Visuals pure helpers (URDF <visual> rendering) ----

describe("Visuals helpers — composeWorld / primitiveSpec", () => {
  it("composeWorld multiplies frame · origin (origin is FRAME-local)", () => {
    // frame: +90° about Z, sitting at (1,0,0); origin: (0,2,0) in frame space.
    const frame = new THREE.Matrix4().makeRotationZ(Math.PI / 2).setPosition(1, 0, 0);
    const origin = new THREE.Matrix4().makeTranslation(0, 2, 0);
    const p = new THREE.Vector3().setFromMatrixPosition(
      composeWorld(frame.toArray(), origin.toArray()),
    );
    // frame-local +Y lands on world -X under Rz(90°) → (1-2, 0, 0).
    expect(p.x).toBeCloseTo(-1, 12);
    expect(p.y).toBeCloseTo(0, 12);
    expect(p.z).toBeCloseTo(0, 12);
  });

  it("primitiveSpec maps DTO size fields to three constructor args", () => {
    const base: Omit<VisualInfo, "kind"> = {
      frame: 0,
      origin: new THREE.Matrix4().toArray(),
      halfExtents: null,
      radius: null,
      length: null,
      color: null,
      meshPath: null,
      meshScale: null,
      raw: null,
    };
    // engine ships HALF-extents; three's BoxGeometry takes full extents.
    const box = primitiveSpec({ ...base, kind: "box", halfExtents: [0.1, 0.2, 0.3] });
    expect(box).toEqual({ shape: "box", args: [0.2, 0.4, 0.6], zAligned: false });
    // URDF cylinders are Z-aligned → need the +90° X wrapper (three is Y-aligned).
    const cyl = primitiveSpec({ ...base, kind: "cylinder", radius: 0.04, length: 0.3 });
    expect(cyl?.shape).toBe("cylinder");
    expect(cyl?.args.slice(0, 3)).toEqual([0.04, 0.04, 0.3]);
    expect(cyl?.zAligned).toBe(true);
    // meshes carry no primitive spec (loaded asynchronously instead).
    expect(primitiveSpec({ ...base, kind: "mesh" })).toBeNull();
  });
});
