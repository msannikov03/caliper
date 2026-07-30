// Headless unit tests for the verdict viewers' pure logic.
//
// Every positive case runs against a REAL document: the fixtures in ./fixtures
// are the verbatim stdout of `caliper-learn {eval,debug,profile,autopsy} --json`
// (and eval.to_json for the sweep ranking), generated against the repo's own
// showcase6 / gripper_arm fixtures and a tiny ACT checkpoint. If the python
// reporters change shape, these tests are what notices.
//
// Coverage:
//  - detection of all four kinds by structural fingerprint, on real files
//  - the autopsy fingerprint wins over the eval/profile ones it CONTAINS
//  - additive unknown fields are tolerated; structurally wrong docs are
//    rejected with a plain-English reason (never half-rendered)
//  - the sweep ranking is refused with the hint that names it
//  - Wilson bar geometry, including the 0/N and N/N edges
//  - findings normalization across the four reporters' anchor fields
//  - latency row prep: canonical stage order, over-budget flag, chunk rows

import { describe, expect, it } from "vitest";

import {
  chunkRows,
  chunkSummary,
  datasetMetrics,
  detectVerdict,
  evalHeadline,
  evalInterval,
  evalMetrics,
  findingCounts,
  fmtMs,
  fmtNum,
  fmtPct,
  latencyMetrics,
  latencyRows,
  normalizeFindings,
  sevChip,
  successCriterion,
  verdictLabel,
  wilsonBar,
} from "./verdicts";
import type {
  AutopsyDoc,
  DebugDoc,
  Detection,
  EvalDoc,
  ProfileDoc,
  VerdictKind,
} from "./verdicts";

import autopsyFixture from "./fixtures/autopsy.json";
import debugFixture from "./fixtures/debug.json";
import evalFixture from "./fixtures/eval.json";
import evalTaskFixture from "./fixtures/eval_task.json";
import profileFixture from "./fixtures/profile.json";
import profileOverBudgetFixture from "./fixtures/profile_over_budget.json";
import sweepFixture from "./fixtures/sweep.json";

/** A mutable deep copy — the imported fixtures are shared across tests. */
function clone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T;
}

/** Detect, assert the kind, and hand back that kind's typed document. One
 *  helper per kind: a single generic one cannot preserve the union's
 *  kind↔doc correlation through its return type. */
function asEval(json: unknown): EvalDoc {
  const d = detectVerdict(json);
  if (d.kind !== "eval") throw new Error(mismatch("eval", d));
  return d.doc;
}
function asAutopsy(json: unknown): AutopsyDoc {
  const d = detectVerdict(json);
  if (d.kind !== "autopsy") throw new Error(mismatch("autopsy", d));
  return d.doc;
}
function asProfile(json: unknown): ProfileDoc {
  const d = detectVerdict(json);
  if (d.kind !== "profile") throw new Error(mismatch("profile", d));
  return d.doc;
}
function asDebug(json: unknown): DebugDoc {
  const d = detectVerdict(json);
  if (d.kind !== "debug") throw new Error(mismatch("debug", d));
  return d.doc;
}

function mismatch(want: VerdictKind, got: Detection): string {
  return `expected ${want}, got ${got.kind}${got.kind === "unknown" ? `: ${got.why}` : ""}`;
}

/** Detect and assert the rejection, returning the reason. */
function rejected(json: unknown): string {
  const d = detectVerdict(json);
  if (d.kind !== "unknown") throw new Error(`expected a rejection, got ${d.kind}`);
  return d.why;
}

// ---- detection on the real documents ----

describe("detectVerdict — the four real caliper-learn documents", () => {
  it("recognizes `eval --json` and keeps its numbers intact", () => {
    const doc = asEval(evalFixture);
    expect([doc.n_success, doc.n_episodes]).toEqual([3, 6]);
    expect(doc.episodes).toHaveLength(6);
    // the seeds are the report's reproduction handles — every one is present
    expect(doc.episodes.map((e) => e.seed)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(doc.findings.map((f) => f.code)).toEqual(["E002"]);
  });

  it("recognizes a TASK-scored eval, criterion sentence and all", () => {
    const doc = asEval(evalTaskFixture);
    expect([doc.n_success, doc.n_episodes]).toEqual([0, 3]);
    expect(successCriterion(doc)).toContain("cube's center is inside");
    expect(doc.findings.map((f) => f.code)).toEqual(["E001", "E003"]);
  });

  it("recognizes `debug --json` (the P-findings payload)", () => {
    const doc = asDebug(debugFixture);
    expect(doc.policy_dir).toMatch(/ckpt_zero$/);
    expect(new Set(doc.findings.map((f) => f.code))).toEqual(new Set(["P001", "P002", "P006"]));
  });

  it("recognizes `profile --json`, chunk stats included", () => {
    const doc = asProfile(profileFixture);
    expect(doc.fps).toBe(200);
    expect(Object.keys(doc.stages).sort()).toEqual(["inference", "obs_build", "step", "total"]);
    expect(doc.chunk?.source).toBe("config");
    expect(doc.chunk?.period).toBe(4);
    expect(doc.findings).toEqual([]);
  });

  it("recognizes an over-budget profile (a real L001)", () => {
    const doc = asProfile(profileOverBudgetFixture);
    expect(doc.findings.map((f) => f.code)).toEqual(["L001"]);
    expect(doc.achievable_hz).toBeLessThan(doc.fps);
  });

  it("recognizes `autopsy --json` — and the autopsy fingerprint WINS over the eval and profile it contains", () => {
    const doc = asAutopsy(autopsyFixture);
    expect(doc.verdict).toContain("closed-loop");
    expect(new Set(doc.dataset.findings.map((f) => f.code))).toEqual(
      new Set(["D001", "D004", "D009", "D011"]),
    );
    expect(new Set(doc.policy_findings.map((f) => f.code))).toEqual(
      new Set(["P001", "P002", "P004", "P006"]),
    );
    // both optional sections ran here
    expect(doc.eval?.n_episodes).toBe(4);
    expect(doc.latency?.fps).toBe(25);
  });

  it("accepts an autopsy whose E and L sections are null (no robot+task)", () => {
    const doc = clone(autopsyFixture) as Record<string, unknown>;
    doc.eval = null;
    doc.latency = null;
    const out = asAutopsy(doc);
    expect(out.eval).toBeNull();
    expect(out.latency).toBeNull();
  });

  it("labels every kind", () => {
    expect(verdictLabel("eval")).toContain("(E)");
    expect(verdictLabel("debug")).toContain("(P)");
    expect(verdictLabel("profile")).toContain("(L)");
    expect(verdictLabel("autopsy")).toContain("D");
  });
});

// ---- additive tolerance (the stability contract) ----

describe("detectVerdict — additive fields are tolerated", () => {
  it("opens an eval report that grew new fields at every level", () => {
    const doc = clone(evalFixture) as Record<string, unknown>;
    doc.wilson_z = 1.96; // a future top-level field
    (doc.episodes as Record<string, unknown>[])[0].contact_forces = [1, 2, 3];
    (doc.findings as Record<string, unknown>[])[0].doc_url = "https://example/E002";
    const out = asEval(doc);
    expect(out.n_success).toBe(3);
    expect(normalizeFindings(out.findings)[0].code).toBe("E002");
  });

  it("opens a profile whose stage set grew, and shows the new stage", () => {
    const doc = clone(profileFixture) as ProfileDoc;
    doc.stages.postprocess = { p50: 0.001, p95: 0.002, p99: 0.003, max: 0.004 };
    const out = asProfile(doc);
    // canonical stages keep their order; the newcomer is appended, not dropped
    expect(latencyRows(out).map((r) => r.stage)).toEqual([
      "obs_build",
      "inference",
      "step",
      "total",
      "postprocess",
    ]);
  });

  it("opens an eval report from before success_criterion existed", () => {
    const doc = clone(evalFixture) as Record<string, unknown>;
    delete doc.success_criterion;
    const out = asEval(doc);
    expect(successCriterion(out)).toBe("not recorded by this report");
  });

  it("says what a null criterion means rather than leaving it blank", () => {
    expect(successCriterion(evalFixture as unknown as EvalDoc)).toContain("termination_fn");
  });
});

// ---- rejection with a reason ----

describe("detectVerdict — rejections name the problem", () => {
  it("names a sweep ranking instead of half-opening it", () => {
    const why = rejected(sweepFixture);
    expect(why).toContain("SWEEP");
    expect(why).toContain("single run");
  });

  it("rejects a plain JSON array, a scalar and null", () => {
    expect(rejected([1, 2, 3])).toContain("array");
    expect(rejected("eval")).toContain("a string");
    expect(rejected(null)).toContain("null");
    expect(rejected(42)).toContain("a number");
  });

  it("lists the keys it found when nothing fingerprints", () => {
    const why = rejected({ hello: 1, world: 2 });
    expect(why).toContain("no caliper-learn verdict fingerprint");
    expect(why).toContain("hello, world");
    expect(rejected({})).toContain("no keys at all");
  });

  it("rejects an eval report whose episodes are not episodes", () => {
    const doc = clone(evalFixture) as Record<string, unknown>;
    doc.episodes = "6";
    expect(rejected(doc)).toBe("the eval report.episodes must be a list, got a string");
  });

  it("points at the exact episode field that is wrong", () => {
    const doc = clone(evalFixture) as Record<string, unknown>;
    (doc.episodes as Record<string, unknown>[])[2].steps = null;
    expect(rejected(doc)).toBe("the eval report.episodes[2].steps must be a number, got null");
  });

  it("rejects a truncated eval report (a missing headline number)", () => {
    const doc = clone(evalFixture) as Record<string, unknown>;
    delete doc.ci95_low;
    expect(rejected(doc)).toContain("ci95_low must be a number, got missing");
  });

  it("rejects a profile whose percentiles are strings", () => {
    const doc = clone(profileFixture) as Record<string, unknown>;
    (doc.stages as Record<string, Record<string, unknown>>).total.p95 = "3.6ms";
    expect(rejected(doc)).toBe("the latency profile.stages.total.p95 must be a number, got a string");
  });

  it("rejects a profile whose chunk block is malformed", () => {
    const doc = clone(profileFixture) as Record<string, unknown>;
    (doc.chunk as Record<string, unknown>).refill = 0.001;
    expect(rejected(doc)).toContain("chunk.refill must be a stats object");
  });

  it("rejects a debug payload with no findings list", () => {
    expect(rejected({ policy_dir: "/ckpt", findings: {} })).toContain(
      "findings must be a list of findings",
    );
    expect(rejected({ policy_dir: 7, findings: [] })).toContain("policy_dir must be a string");
  });

  it("rejects a finding that is missing its stable code", () => {
    const doc = clone(debugFixture) as Record<string, unknown>;
    delete (doc.findings as Record<string, unknown>[])[0].code;
    expect(rejected(doc)).toBe(
      "the policy-debugger report.findings[0].code must be a string, got missing",
    );
  });

  it("rejects an autopsy whose dataset section is not a dataset section", () => {
    const doc = clone(autopsyFixture) as Record<string, unknown>;
    (doc.dataset as Record<string, unknown>).total_frames = "128";
    expect(rejected(doc)).toBe("the autopsy report.dataset.total_frames must be a number, got a string");
  });

  it("rejects an autopsy whose embedded eval is broken, naming the section", () => {
    const doc = clone(autopsyFixture) as Record<string, unknown>;
    (doc.eval as Record<string, unknown>).success_rate = "0%";
    expect(rejected(doc)).toBe("the autopsy report.eval.success_rate must be a number, got a string");
  });

  it("rejects an autopsy whose embedded latency profile is broken", () => {
    const doc = clone(autopsyFixture) as Record<string, unknown>;
    (doc.latency as Record<string, unknown>).stages = [];
    expect(rejected(doc)).toBe("the autopsy report.latency.stages must be an object, got an array");
  });
});

// ---- Wilson bar geometry ----

describe("wilsonBar", () => {
  it("places the real eval report's band and point", () => {
    const doc = evalFixture as unknown as EvalDoc;
    const bar = wilsonBar(doc.ci95_low, doc.ci95_high, doc.success_rate);
    expect(bar.lowPct).toBeCloseTo(18.76, 2);
    expect(bar.highPct).toBeCloseTo(81.24, 2);
    expect(bar.widthPct).toBeCloseTo(62.48, 2);
    expect(bar.pointPct).toBe(50);
    // a 6-episode run is a very wide band — the panel must be able to say so
    expect(bar.widthPct).toBeGreaterThan(60);
  });

  it("pins the 0/N edge to the left wall without a negative width", () => {
    const doc = evalTaskFixture as unknown as EvalDoc;
    const bar = wilsonBar(doc.ci95_low, doc.ci95_high, doc.success_rate);
    expect(bar.lowPct).toBe(0);
    expect(bar.pointPct).toBe(0);
    expect(bar.highPct).toBeCloseTo(56.15, 2);
    expect(bar.widthPct).toBeCloseTo(56.15, 2);
  });

  it("pins the N/N edge to the right wall", () => {
    const bar = wilsonBar(0.5407, 1, 1);
    expect(bar.highPct).toBe(100);
    expect(bar.pointPct).toBe(100);
    expect(bar.lowPct + bar.widthPct).toBeCloseTo(100, 6);
  });

  it("clamps out-of-range and non-finite inputs instead of painting outside the track", () => {
    expect(wilsonBar(-0.2, 1.4, 1.2)).toEqual({
      lowPct: 0,
      highPct: 100,
      widthPct: 100,
      pointPct: 100,
    });
    expect(wilsonBar(NaN, NaN, NaN)).toEqual({
      lowPct: 0,
      highPct: 0,
      widthPct: 0,
      pointPct: 0,
    });
  });

  it("collapses an inverted interval rather than inventing a range", () => {
    const bar = wilsonBar(0.8, 0.2, 0.5);
    expect(bar.lowPct).toBe(80);
    expect(bar.highPct).toBe(80);
    expect(bar.widthPct).toBe(0);
    expect(bar.pointPct).toBe(50);
  });
});

// ---- findings normalization ----

describe("normalizeFindings / sevChip / findingCounts", () => {
  it("carries the policy debugger's anchors into ref chips", () => {
    const doc = asAutopsy(autopsyFixture);
    const findings = normalizeFindings(doc.policy_findings);
    const p001 = findings.find((f) => f.code === "P001");
    expect(p001?.severity).toBe("error");
    expect(p001?.suggestion).toContain("zeroed/corrupted weights");
    expect(p001?.refs).toEqual(["value 0"]); // checkpoint-wide: no dof
    const p002 = findings.find((f) => f.code === "P002");
    expect(p002?.refs).toContain("dof 0");
  });

  it("carries the dataset doctor's episode/frame/dof/feature anchors", () => {
    const doc = asAutopsy(autopsyFixture);
    const findings = normalizeFindings(doc.dataset.findings);
    const d001 = findings.find((f) => f.code === "D001");
    expect(d001?.refs).toEqual(["dof 1", "action"]);
    // D011 localizes an instant: episode AND frame, in that order
    const d011 = findings.find((f) => f.code === "D011");
    expect(d011?.refs).toEqual(["ep 0", "frame 35"]);
  });

  it("leaves eval/profile findings ref-free and keeps the fix hint", () => {
    const doc = asEval(evalFixture);
    const [e002] = normalizeFindings(doc.findings);
    expect(e002.refs).toEqual([]);
    expect(e002.suggestion).toContain("n_episodes");
  });

  it("treats a missing or empty fix hint as no suggestion", () => {
    const [a, b] = normalizeFindings([
      { code: "X001", severity: "warn", message: "m" },
      { code: "X002", severity: "warn", message: "m", fix_hint: "" },
    ]);
    expect(a.suggestion).toBeNull();
    expect(b.suggestion).toBeNull();
  });

  it("maps both severity spellings onto the shared chips", () => {
    // the Rust dataset doctor says "warning", the python reporters say "warn"
    expect(sevChip("warning")).toEqual({ cls: "sev-warn", label: "WARN" });
    expect(sevChip("warn")).toEqual({ cls: "sev-warn", label: "WARN" });
    expect(sevChip("error")).toEqual({ cls: "sev-error", label: "ERR" });
    expect(sevChip("info")).toEqual({ cls: "sev-info", label: "INFO" });
    // a severity we do not know keeps its own text — it must not READ as ours
    expect(sevChip("critical")).toEqual({ cls: "sev-warn", label: "CRITIC" });
    expect(sevChip("")).toEqual({ cls: "sev-warn", label: "?" });
  });

  it("tallies the real autopsy sections", () => {
    const doc = asAutopsy(autopsyFixture);
    // the dataset section: 4 warnings + 1 info, no errors
    expect(findingCounts(doc.dataset.findings)).toEqual({ errors: 0, warnings: 4, infos: 1 });
    // the policy section leads with errors (P001 + three P004 rows)
    expect(findingCounts(doc.policy_findings).errors).toBe(4);
    expect(findingCounts([])).toEqual({ errors: 0, warnings: 0, infos: 0 });
  });
});

// ---- latency rows ----

describe("latencyRows / chunkRows", () => {
  it("orders the real profile's stages the way the tick is built", () => {
    const doc = asProfile(profileFixture);
    expect(latencyRows(doc).map((r) => r.stage)).toEqual([
      "obs_build",
      "inference",
      "step",
      "total",
    ]);
    // ... even when the document stored them sorted (the autopsy's to_json does)
    const embedded = (asAutopsy(autopsyFixture)).latency as ProfileDoc;
    expect(Object.keys(embedded.stages)).toEqual(["inference", "obs_build", "step", "total"]);
    expect(latencyRows(embedded).map((r) => r.stage)).toEqual([
      "obs_build",
      "inference",
      "step",
      "total",
    ]);
  });

  it("converts to milliseconds and flags only the stages over the deadline", () => {
    const ok = asProfile(profileFixture);
    expect(ok.budget_s).toBe(0.005);
    expect(ok.stages.total.p95).toBeLessThan(ok.budget_s);
    expect(latencyRows(ok).every((r) => !r.overBudget)).toBe(true);
    expect(latencyRows(ok).find((r) => r.stage === "total")?.p95Ms).toBe("3.601");

    const over = asProfile(profileOverBudgetFixture);
    const total = latencyRows(over).find((r) => r.stage === "total");
    expect(total?.overBudget).toBe(true);
  });

  it("splits the chunked policy's refill from its pops", () => {
    const doc = asProfile(profileFixture);
    const rows = chunkRows(doc);
    expect(rows?.map((r) => r.stage)).toEqual(["refill", "pop"]);
    expect(chunkSummary(doc.chunk!)).toBe("config, every 4 ticks · 10 refill ticks");
  });

  it("has no chunk rows for an unchunked profile, and drops `pop` when every tick refilled", () => {
    const noChunk = clone(profileFixture) as ProfileDoc;
    noChunk.chunk = null;
    expect(chunkRows(noChunk)).toBeNull();

    const allRefill = clone(profileFixture) as ProfileDoc;
    allRefill.chunk!.pop = null;
    allRefill.chunk!.period = null;
    expect(chunkRows(allRefill)?.map((r) => r.stage)).toEqual(["refill"]);
    expect(chunkSummary(allRefill.chunk!)).toContain("period undetermined");
  });
});

// ---- readouts ----

describe("headline / metric formatting", () => {
  it("never states a rate without the counts behind it", () => {
    const doc = asEval(evalFixture);
    expect(evalHeadline(doc)).toBe("3 / 6 episodes succeeded");
    expect(evalInterval(doc)).toBe("50.0% · 95% CI [18.8%, 81.2%]");
    const one = clone(doc);
    one.n_episodes = 1;
    one.n_success = 1;
    expect(evalHeadline(one)).toBe("1 / 1 episode succeeded");
  });

  it("builds the eval, latency and dataset readout rows", () => {
    const ev = asEval(evalFixture);
    expect(evalMetrics(ev)).toEqual([
      ["mean return", "-2.731"],
      ["median return", "-2.551"],
      ["steps to success", "16.3"],
      ["episodes", "6"],
    ]);
    const pr = asProfile(profileFixture);
    expect(latencyMetrics(pr)).toEqual([
      ["rate", "200 Hz"],
      ["budget", "5.000 ms"],
      ["achievable", "278 Hz"],
      ["ticks", "40"],
      ["jitter", "0.426 ms"],
      ["over budget", "0%"],
    ]);
    const ap = asAutopsy(autopsyFixture);
    expect(datasetMetrics(ap.dataset)).toEqual([
      ["episodes", "4"],
      ["frames", "128"],
      ["fps", "25"],
    ]);
  });

  it("prints an em dash for values a report legitimately has none of", () => {
    const doc = clone(evalFixture) as EvalDoc;
    doc.mean_steps_to_success = null; // zero successes: there is no such mean
    expect(evalMetrics(doc)[2]).toEqual(["steps to success", "—"]);
    expect(fmtNum(null)).toBe("—");
    expect(fmtNum(undefined)).toBe("—");
    expect(fmtMs(null)).toBe("—");
    expect(fmtPct(NaN)).toBe("—");
    expect(fmtPct(0.5)).toBe("50.0%");
    expect(fmtMs(0.0025601979)).toBe("2.560");
  });
});
