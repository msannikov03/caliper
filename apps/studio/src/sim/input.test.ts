// Headless unit tests for the pure live-input helpers (src/sim/input.ts):
// the per-frame jog integrator and its limit clamp, the analog deadband +
// cubic response curve, gamepad button EDGE detection, and the keymap
// decision App defers to.
//
// Everything here is frame-local arithmetic — the numbers are asserted
// exactly, because a drifting jog rate or a mis-signed stick axis is the kind
// of bug that only shows up with a human's hand on the controls.

import { describe, it, expect } from "vitest";
import {
  applyJog,
  applyTipVelocity,
  axisResponse,
  clampJoint,
  gamepadIntent,
  isJogKey,
  jogDelta,
  jogDirection,
  jogRate,
  liveKeyAction,
  stepJoint,
  GAMEPAD_DEADBAND,
  JOG_RATE_PRISMATIC,
  JOG_RATE_REVOLUTE,
  TIP_RATE,
} from "./input";

const FRAME = 1 / 60;

describe("jog rate + delta — one frame of held-key motion", () => {
  it("uses the kind's base rate when the model has no velocity limit", () => {
    expect(jogRate("revolute")).toBe(JOG_RATE_REVOLUTE);
    expect(jogRate("prismatic")).toBe(JOG_RATE_PRISMATIC);
    expect(jogRate("revolute", null)).toBe(1.5);
  });

  it("caps at the joint's velocity limit but never speeds up past the base", () => {
    expect(jogRate("revolute", 0.5)).toBe(0.5);
    expect(jogRate("revolute", 9)).toBe(1.5);
    expect(jogRate("prismatic", 0.1)).toBeCloseTo(0.1, 12);
    expect(jogRate("revolute", 0)).toBe(1.5); // 0 = "unbounded", not "frozen"
  });

  it("integrates rate × dt with the sign of the direction", () => {
    expect(jogDelta("revolute", FRAME, 1)).toBeCloseTo(0.025, 12);
    expect(jogDelta("revolute", FRAME, -1)).toBeCloseTo(-0.025, 12);
    expect(jogDelta("prismatic", FRAME, 1)).toBeCloseTo(0.25 / 60, 12);
    expect(jogDelta("revolute", FRAME, 1, 0.6)).toBeCloseTo(0.01, 12);
  });

  it("contributes nothing with no direction or a stalled/rewound clock", () => {
    expect(jogDelta("revolute", FRAME, 0)).toBe(0);
    expect(jogDelta("revolute", 0, 1)).toBe(0);
    expect(jogDelta("revolute", -0.02, 1)).toBe(0);
    expect(jogDelta("revolute", FRAME, NaN)).toBe(0);
  });
});

describe("clampJoint / applyJog — the target never leaves the limits", () => {
  it("clamps into an explicit limit", () => {
    expect(clampJoint(2, [-1, 1])).toBe(1);
    expect(clampJoint(-2, [-1, 1])).toBe(-1);
    expect(clampJoint(0.4, [-1, 1])).toBe(0.4);
  });

  it("falls back to the slider's own span for an unbounded joint", () => {
    expect(clampJoint(4, null)).toBe(Math.PI);
    expect(clampJoint(-4, undefined)).toBe(-Math.PI);
    expect(clampJoint(2, null, "prismatic")).toBe(0.5);
    expect(clampJoint(-2, null, "prismatic")).toBe(-0.5);
  });

  it("folds deltas in and clamps only what moved", () => {
    const out = applyJog([0, 0], [0.1, 0], [[-0.05, 0.05], null], ["revolute", "revolute"]);
    expect(out).toEqual([0.05, 0]);
  });

  it("leaves untouched joints alone even when they sit outside their limit", () => {
    // a pose that arrived out of range (a loosened URDF, a solver overshoot)
    // is not this function's business — only the joint being jogged is
    const out = applyJog([9, 0], [0, 0.25], [[-1, 1], [-1, 1]]);
    expect(out).toEqual([9, 0.25]);
  });

  it("returns a fresh array and never mutates the input target", () => {
    const target = [0.1, 0.2];
    const out = applyJog(target, [0.01, 0], [null, null]);
    expect(target).toEqual([0.1, 0.2]);
    expect(out).not.toBe(target);
  });
});

describe("axisResponse — deadband + cubic curve", () => {
  it("reads exactly zero inside the deadband (a resting stick cannot creep)", () => {
    expect(axisResponse(0)).toBe(0);
    expect(axisResponse(0.1)).toBe(0);
    expect(axisResponse(GAMEPAD_DEADBAND)).toBe(0);
    expect(axisResponse(-GAMEPAD_DEADBAND)).toBe(0);
    expect(axisResponse(NaN)).toBe(0);
  });

  it("still reaches ±1 at full deflection", () => {
    expect(axisResponse(1)).toBe(1);
    expect(axisResponse(-1)).toBe(-1);
    expect(axisResponse(1.4)).toBe(1); // over-range hardware clamps, not blows up
  });

  it("cubes the re-normalized remainder, sign preserved", () => {
    // 0.575 → (0.575 − 0.15)/0.85 = 0.5 → 0.5³ = 0.125
    expect(axisResponse(0.575)).toBeCloseTo(0.125, 12);
    expect(axisResponse(-0.575)).toBeCloseTo(-0.125, 12);
    // half deflection stays gentle: well under half speed
    expect(axisResponse(0.5)).toBeLessThan(0.1);
  });
});

describe("gamepadIntent — sticks to tip velocity, buttons to edges", () => {
  const none: boolean[] = [];

  it("maps the sticks to world X/Y/Z with the vertical axes inverted", () => {
    const i = gamepadIntent([1, -1, 0, -1], none, none);
    expect(i.tip).toEqual([TIP_RATE, TIP_RATE, TIP_RATE]);
    expect(i.moving).toBe(true);
  });

  it("pushing the sticks the other way reverses every axis", () => {
    const i = gamepadIntent([-1, 1, 0, 1], none, none);
    expect(i.tip).toEqual([-TIP_RATE, -TIP_RATE, -TIP_RATE]);
  });

  it("is idle (and not moving) with every stick inside the deadband", () => {
    const i = gamepadIntent([0.1, -0.12, 0.9, 0.14], none, none);
    expect(i.tip).toEqual([0, 0, 0]);
    expect(i.moving).toBe(false);
  });

  it("survives a pad that reports fewer axes than we read", () => {
    const i = gamepadIntent([0.5], none, none);
    expect(i.tip[1]).toBe(0);
    expect(i.tip[2]).toBe(0);
  });

  it("fires A/B exactly once per press, not once per frame held", () => {
    const down = gamepadIntent([], [true, true], [false, false]);
    expect(down.togglePause).toBe(true);
    expect(down.reset).toBe(true);
    const held = gamepadIntent([], [true, true], [true, true]);
    expect(held.togglePause).toBe(false);
    expect(held.reset).toBe(false);
    const release = gamepadIntent([], [false, false], [true, true]);
    expect(release.togglePause).toBe(false);
    // a press after the release is a fresh edge
    expect(gamepadIntent([], [true, false], [false, false]).togglePause).toBe(true);
    expect(release.reset).toBe(false);
  });

  it("treats a first-ever poll (no previous frame) as a press edge", () => {
    expect(gamepadIntent([], [true], []).togglePause).toBe(true);
  });
});

describe("applyTipVelocity — cartesian goal integrator", () => {
  it("advances the goal by velocity × dt", () => {
    const out = applyTipVelocity([0.3, 0, 0.2], [0.25, 0, -0.25], 0.1);
    expect(out[0]).toBeCloseTo(0.325, 12);
    expect(out[1]).toBeCloseTo(0, 12);
    expect(out[2]).toBeCloseTo(0.175, 12);
  });

  it("stands still on a non-positive dt", () => {
    expect(applyTipVelocity([1, 2, 3], [1, 1, 1], 0)).toEqual([1, 2, 3]);
    expect(applyTipVelocity([1, 2, 3], [1, 1, 1], -1)).toEqual([1, 2, 3]);
  });
});

describe("keyboard — direction, selection, and what a key means", () => {
  it("nets the held keys, with opposing keys cancelling", () => {
    expect(jogDirection(["="])).toBe(1);
    expect(jogDirection(["+"])).toBe(1);
    expect(jogDirection(["ArrowUp"])).toBe(1);
    expect(jogDirection(["-"])).toBe(-1);
    expect(jogDirection(["ArrowDown"])).toBe(-1);
    expect(jogDirection(new Set(["=", "-"]))).toBe(0);
    expect(jogDirection([])).toBe(0);
    expect(jogDirection(["q"])).toBe(0);
  });

  it("knows which keys it has to watch for a keyup", () => {
    expect(isJogKey("=")).toBe(true);
    expect(isJogKey("ArrowDown")).toBe(true);
    expect(isJogKey("[")).toBe(false);
    expect(isJogKey(" ")).toBe(false);
  });

  it("wraps the joint selection at both ends", () => {
    expect(stepJoint(0, 1, 3)).toBe(1);
    expect(stepJoint(2, 1, 3)).toBe(0);
    expect(stepJoint(0, -1, 3)).toBe(2);
    expect(stepJoint(0, 0, 3)).toBe(0);
  });

  it("pins to 0 for a model with no joints or a nonsense selection", () => {
    expect(stepJoint(0, 1, 0)).toBe(0);
    expect(stepJoint(NaN, 1, 4)).toBe(1);
  });

  it("classifies the drive keys and ignores everything else", () => {
    expect(liveKeyAction(" ")).toBe("freeze");
    expect(liveKeyAction("Spacebar")).toBe("freeze"); // older webviews
    expect(liveKeyAction("[")).toBe("prev-joint");
    expect(liveKeyAction("]")).toBe("next-joint");
    expect(liveKeyAction("=")).toBe("jog");
    expect(liveKeyAction("-")).toBe("jog");
    expect(liveKeyAction("ArrowUp")).toBe("jog");
    expect(liveKeyAction("k")).toBeNull();
    expect(liveKeyAction("Escape")).toBeNull();
  });
});
