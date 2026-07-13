// Headless unit tests for the first-run tour's step machine (src/tour.ts).
//
// Everything here is pure: no DOM queries, no React, no real localStorage —
// tourReducer gets plain numbers and shouldShowTour/markTourDone get fake
// (including deliberately broken) storage objects.

import { describe, it, expect } from "vitest";
import {
  TOUR_KEY,
  TOUR_STEPS,
  anchorSelector,
  clampStep,
  isLastStep,
  markTourDone,
  shouldShowTour,
  stepProgress,
  tourReducer,
} from "./tour";
import type { TourFlagStore } from "./tour";

function fakeStore(initial: Record<string, string> = {}): TourFlagStore & {
  data: Record<string, string>;
} {
  const data = { ...initial };
  return {
    data,
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => {
      data[k] = v;
    },
  };
}

/** A storage whose every method throws (private mode / quota exceeded). */
const brokenStore: TourFlagStore = {
  getItem() {
    throw new Error("storage unavailable");
  },
  setItem() {
    throw new Error("storage unavailable");
  },
};

describe("TOUR_STEPS", () => {
  it("has at most 6 steps, with unique non-empty ids and titles", () => {
    expect(TOUR_STEPS.length).toBeGreaterThan(0);
    expect(TOUR_STEPS.length).toBeLessThanOrEqual(6);
    const ids = TOUR_STEPS.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const s of TOUR_STEPS) {
      expect(s.id).not.toBe("");
      expect(s.title).not.toBe("");
      expect(s.body).not.toBe("");
    }
  });

  it("covers the open button, every mode-tab anchor it names, and ⌘K", () => {
    const anchors = TOUR_STEPS.map((s) => s.anchor);
    expect(anchors).toContain("open");
    expect(anchors).toContain("tab-jog");
    expect(anchors).toContain("tab-motion");
    expect(anchors).toContain("tab-simulate");
    expect(anchors).toContain("tab-graph");
    expect(anchors).toContain("palette");
  });

  it("anchorSelector builds a [data-tour] selector, or null for centered steps", () => {
    expect(anchorSelector({ id: "x", anchor: "open", title: "t", body: "b" })).toBe(
      '[data-tour="open"]',
    );
    expect(anchorSelector({ id: "x", anchor: null, title: "t", body: "b" })).toBeNull();
  });
});

describe("tourReducer", () => {
  it("walks forward through every step and closes after the last", () => {
    let s: number | null = 0;
    for (let k = 1; k < TOUR_STEPS.length; k++) {
      s = tourReducer(s, "next");
      expect(s).toBe(k);
    }
    expect(isLastStep(s as number)).toBe(true);
    expect(tourReducer(s, "next")).toBeNull(); // Done
  });

  it("back steps to the previous step and clamps at the first", () => {
    expect(tourReducer(2, "back")).toBe(1);
    expect(tourReducer(1, "back")).toBe(0);
    expect(tourReducer(0, "back")).toBe(0); // clamp, not close
  });

  it("skip closes from any step", () => {
    for (let i = 0; i < TOUR_STEPS.length; i++) {
      expect(tourReducer(i, "skip")).toBeNull();
    }
  });

  it("null (closed) absorbs every action", () => {
    expect(tourReducer(null, "next")).toBeNull();
    expect(tourReducer(null, "back")).toBeNull();
    expect(tourReducer(null, "skip")).toBeNull();
  });

  it("clamps out-of-range and non-finite state instead of crashing", () => {
    expect(tourReducer(999, "next")).toBeNull(); // clamped to last → Done
    expect(tourReducer(-5, "back")).toBe(0);
    expect(tourReducer(Number.NaN, "back")).toBe(0);
    expect(clampStep(2.9)).toBe(2); // fractional input truncates
  });
});

describe("progress / last-step helpers", () => {
  it("stepProgress renders 1-based n / total", () => {
    expect(stepProgress(0)).toBe(`1 / ${TOUR_STEPS.length}`);
    expect(stepProgress(TOUR_STEPS.length - 1)).toBe(
      `${TOUR_STEPS.length} / ${TOUR_STEPS.length}`,
    );
  });

  it("isLastStep is true only on the final step", () => {
    expect(isLastStep(0)).toBe(false);
    expect(isLastStep(TOUR_STEPS.length - 1)).toBe(true);
  });
});

describe("shouldShowTour / markTourDone", () => {
  it("shows on a fresh store, and never again once marked done", () => {
    const store = fakeStore();
    expect(shouldShowTour(store)).toBe(true);
    markTourDone(store);
    expect(store.data[TOUR_KEY]).toBe("1");
    expect(shouldShowTour(store)).toBe(false);
  });

  it("respects a pre-existing flag regardless of its value", () => {
    expect(shouldShowTour(fakeStore({ [TOUR_KEY]: "1" }))).toBe(false);
    expect(shouldShowTour(fakeStore({ [TOUR_KEY]: "true" }))).toBe(false);
  });

  it("never auto-shows when storage is broken (never-nag fallback)", () => {
    expect(shouldShowTour(brokenStore)).toBe(false);
  });

  it("markTourDone swallows storage failures", () => {
    expect(() => markTourDone(brokenStore)).not.toThrow();
  });
});
