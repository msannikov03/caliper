// Headless tests for the Data-mode pure helpers (episodes.ts) and the palette
// gating of the new dataset commands. No React, no Tauri — same discipline as
// the rest of the vitest harness.
import { describe, it, expect } from "vitest";
import {
  addTag,
  alignChannel,
  clampSplitFrame,
  cursorIndex,
  dimLabels,
  fmtDuration,
  mergePartner,
  removeTag,
  rowView,
  seriesColor,
  SERIES_COLORS,
} from "./episodes";
import { buildCommands } from "../commands";
import type { CommandCtx } from "../commands";

describe("episodes — pure helpers", () => {
  it("dimLabels prefers dataset names only when dimension-consistent", () => {
    expect(dimLabels("q", 2, ["shoulder", "elbow"])).toEqual(["shoulder", "elbow"]);
    expect(dimLabels("q", 3, ["shoulder", "elbow"])).toEqual(["q[0]", "q[1]", "q[2]"]);
    expect(dimLabels("q", 1, null)).toEqual(["q[0]"]);
  });

  it("alignChannel drops length-mismatched dims instead of feeding uPlot garbage", () => {
    const times = [0, 0.1, 0.2];
    const ch = { name: "s", series: [[1, 2, 3], [4, 5], [6, 7, 8]] };
    expect(alignChannel(times, ch)).toEqual([times, [1, 2, 3], [6, 7, 8]]);
  });

  it("fmtDuration formats seconds and minutes, and dashes non-finite", () => {
    expect(fmtDuration(8.44)).toBe("8.4s");
    expect(fmtDuration(83.4)).toBe("1:23.4");
    expect(fmtDuration(61)).toBe("1:01.0"); // sub-10s seconds keep the pad
    expect(fmtDuration(Number.NaN)).toBe("—");
    expect(fmtDuration(-1)).toBe("—");
  });

  it("rowView pre-formats duration and joins tasks", () => {
    const v = rowView({
      index: 3,
      length: 120,
      durationS: 4,
      tasks: ["reach", "hold"],
      tags: ["success"],
    });
    expect(v).toEqual({
      index: 3,
      length: 120,
      duration: "4.0s",
      tasks: "reach · hold",
      tags: ["success"],
    });
  });

  it("mergePartner is adjacent-only and bounded by the episode count", () => {
    expect(mergePartner(0, 3)).toBe(1);
    expect(mergePartner(2, 3)).toBeNull(); // last row has no next
    expect(mergePartner(null, 3)).toBeNull();
    expect(mergePartner(-1, 3)).toBeNull();
  });

  it("clampSplitFrame keeps one frame on each side", () => {
    expect(clampSplitFrame(0, 10)).toBe(1);
    expect(clampSplitFrame(9.6, 10)).toBe(9);
    expect(clampSplitFrame(50, 10)).toBe(9);
    expect(clampSplitFrame(5, 1)).toBeNull(); // too short to split
    expect(clampSplitFrame(Number.POSITIVE_INFINITY, 10)).toBeNull();
  });

  it("cursorIndex maps full-res frames onto the decimated grid, clamped", () => {
    expect(cursorIndex(10, 5, 100)).toBe(2);
    expect(cursorIndex(0, 5, 100)).toBe(0);
    expect(cursorIndex(10_000, 5, 100)).toBe(99);
    expect(cursorIndex(3, 0, 100)).toBe(3); // stride floor of 1
    expect(cursorIndex(3, 5, 0)).toBe(0); // empty series never negative
  });

  it("tag add/remove keep identity on no-ops so callers can skip writes", () => {
    const tags = ["success"];
    expect(addTag(tags, "  ")).toBe(tags);
    expect(addTag(tags, "success")).toBe(tags);
    expect(addTag(tags, " bad-demo ")).toEqual(["success", "bad-demo"]);
    expect(removeTag(tags, "nope")).toBe(tags);
    expect(removeTag(tags, "success")).toEqual([]);
  });

  it("seriesColor cycles the palette and never indexes out of range", () => {
    expect(seriesColor(0)).toBe(SERIES_COLORS[0]);
    expect(seriesColor(SERIES_COLORS.length)).toBe(SERIES_COLORS[0]);
    expect(seriesColor(-1)).toBe(SERIES_COLORS[SERIES_COLORS.length - 1]);
  });
});

describe("palette — dataset command gating", () => {
  const noop = () => {};
  const ctx = (over: Partial<CommandCtx>): CommandCtx => ({
    fixtures: [],
    recents: [],
    poses: [],
    mode: "data",
    robotLoaded: false,
    hasInertia: false,
    urdfPath: null,
    datasetLoaded: false,
    actions: {
      openUrdf: noop,
      openPath: noop,
      setMode: noop,
      planHome: noop,
      planToPose: noop,
      driveHome: noop,
      gravityDrop: noop,
      planRrtHome: noop,
      checkCollision: noop,
      runGraph: noop,
      validateGraph: noop,
      duplicateSelection: noop,
      fitGraphView: noop,
      exportGraph: noop,
      importGraph: noop,
      openDataset: noop,
      refreshDataset: noop,
    },
    ...over,
  });

  it("Open dataset is available even with no robot loaded", () => {
    const open = buildCommands(ctx({})).find((c) => c.id === "data.open");
    expect(open?.enabled).toBe(true);
  });

  it("Refresh dataset is gated on an open dataset", () => {
    const without = buildCommands(ctx({})).find((c) => c.id === "data.refresh");
    expect(without?.enabled).toBe(false);
    expect(without?.hint).toBe("no dataset open");
    const withDs = buildCommands(ctx({ datasetLoaded: true })).find((c) => c.id === "data.refresh");
    expect(withDs?.enabled).toBe(true);
  });
});
