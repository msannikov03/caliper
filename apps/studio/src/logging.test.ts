// Headless unit tests for the crash-log helpers (src/logging.ts).
//
// Pure logic only — no Tauri IPC, no DOM events needed (formatErrorEvent takes
// a structural type so a plain object suffices).

import { describe, it, expect } from "vitest";
import { formatErrorEvent, formatRejection, makeDedupe } from "./logging";

describe("makeDedupe", () => {
  it("allows the first occurrence and suppresses repeats inside the window", () => {
    let t = 0;
    const guard = makeDedupe(1000, () => t);
    expect(guard("boom")).toBe(true);
    expect(guard("boom")).toBe(false); // same tick
    t = 999;
    expect(guard("boom")).toBe(false); // still inside 1 s
    t = 1000;
    expect(guard("boom")).toBe(true); // window elapsed
  });

  it("tracks distinct messages independently", () => {
    let t = 0;
    const guard = makeDedupe(1000, () => t);
    expect(guard("a")).toBe(true);
    expect(guard("b")).toBe(true);
    expect(guard("a")).toBe(false);
    expect(guard("b")).toBe(false);
  });

  it("an error loop yields ~1 log per second, not one per iteration", () => {
    let t = 0;
    const guard = makeDedupe(1000, () => t);
    let sent = 0;
    // 10 ms RAF-style loop for 5 simulated seconds.
    for (let i = 0; i <= 500; i++) {
      t = i * 10;
      if (guard("same error")) sent++;
    }
    expect(sent).toBe(6); // t = 0, 1000, ..., 5000
  });

  it("stays bounded under a flood of unique messages", () => {
    let t = 0;
    const guard = makeDedupe(1000, () => t);
    for (let i = 0; i < 10_000; i++) {
      expect(guard(`err #${i}`)).toBe(true); // unique → never suppressed
    }
    // Still functional after the flood (map was pruned/cleared, not corrupted).
    t = 20_000;
    expect(guard("fresh")).toBe(true);
    expect(guard("fresh")).toBe(false);
  });

  it("prune drops expired entries but keeps live ones", () => {
    let t = 0;
    const guard = makeDedupe(1000, () => t);
    guard("old"); // t=0 — will be expired by prune time
    t = 1500;
    // Crossing the 256-entry threshold triggers a prune pass; "old" (age 1500)
    // is dropped, which keeps the map under the cap WITHOUT the wholesale
    // clear() fallback — so the fresh entries must still dedupe afterwards.
    for (let i = 0; i < 256; i++) guard(`u${i}`);
    expect(guard("u0")).toBe(false); // live entry survived the prune
  });
});

describe("formatErrorEvent", () => {
  it("includes message and source location", () => {
    const s = formatErrorEvent({
      message: "x is not a function",
      filename: "app.js",
      lineno: 12,
      colno: 3,
    });
    expect(s).toBe("uncaught error: x is not a function at app.js:12:3");
  });

  it("appends the Error stack when present", () => {
    const err = new Error("kaput");
    const s = formatErrorEvent({ message: "kaput", error: err });
    expect(s).toContain("uncaught error: kaput");
    expect(s).toContain(err.stack as string);
  });

  it("tolerates a bare event with no fields", () => {
    expect(formatErrorEvent({})).toBe("uncaught error: <no message>");
  });
});

describe("formatRejection", () => {
  it("formats Error reasons with stack", () => {
    const err = new Error("nope");
    const s = formatRejection(err);
    expect(s).toContain("unhandled rejection: nope");
    expect(s).toContain(err.stack as string);
  });

  it("formats string reasons", () => {
    expect(formatRejection("plain")).toBe("unhandled rejection: plain");
  });

  it("JSON-stringifies object reasons", () => {
    expect(formatRejection({ code: 7 })).toBe('unhandled rejection: {"code":7}');
  });

  it("falls back to String() for circular objects", () => {
    const o: Record<string, unknown> = {};
    o.self = o;
    expect(formatRejection(o)).toBe("unhandled rejection: [object Object]");
  });
});
