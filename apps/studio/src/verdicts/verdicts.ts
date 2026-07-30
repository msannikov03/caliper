// ============================================================
// verdicts.ts — pure logic for the verdict viewers: the JSON a
// training run leaves behind (`caliper-learn eval|autopsy|
// profile|debug --json`, or the same objects' python `to_json`)
// parsed into something Data mode can render next to the dataset
// it came from.
//
// Detection is STRUCTURAL — a fingerprint of keys, never the
// filename, because these files get renamed constantly (run7.json,
// before.json, verdict-final-2.json). Unknown ADDITIVE fields are
// tolerated: the stability contract lets these documents grow
// within a minor, and a Studio that refused to open next month's
// eval report would be worse than useless. Structurally WRONG
// documents are rejected with a plain-English reason instead of
// half-rendering.
//
// No React, no Tauri, no store: the vitest suite drives everything
// here against REAL fixture documents in ./fixtures.
// ============================================================

/** The four documents a learning run can leave behind. */
export type VerdictKind = "eval" | "autopsy" | "profile" | "debug";

// ----- wire shapes (snake_case: these come straight from python) -----

/** One row of `eval --json`'s per-episode table. */
export interface EvalEpisodeDoc {
  seed: number;
  success: boolean;
  steps: number;
  episode_return: number;
  /** `task.distance_fn` at the terminal qpos; null when the task has none. */
  final_distance?: number | null;
}

/** A raw finding as any of the four reporters serializes it. The anchor fields
 *  are per-reporter (dataset: feature/episode/dof/frame, policy: feature/dof/
 *  value, eval + profile: none), all optional. */
export interface FindingDoc {
  code: string;
  severity: string;
  message: string;
  fix_hint?: string | null;
  feature?: string | null;
  episode?: number | null;
  frame?: number | null;
  dof?: number | null;
  value?: number | null;
}

/** `caliper-learn eval --json` (eval.EvalResult). */
export interface EvalDoc {
  n_episodes: number;
  n_success: number;
  success_rate: number;
  ci95_low: number;
  ci95_high: number;
  mean_return: number;
  median_return: number;
  mean_steps_to_success?: number | null;
  episodes: EvalEpisodeDoc[];
  findings: FindingDoc[];
  /** The success predicate's sentence when the run was scored by one; null =
   *  "the termination_fn fired". Absent in pre-success-predicate reports. */
  success_criterion?: string | null;
}

/** Wall-time percentiles of one profiled stage, in SECONDS. */
export interface StageStatsDoc {
  p50: number;
  p95: number;
  p99: number;
  max: number;
}

/** Refill-vs-pop split of the inference stage for a chunked policy. */
export interface ChunkStatsDoc {
  source: string; // "config" | "bimodal"
  period?: number | null;
  n_refill_ticks: number;
  refill: StageStatsDoc;
  pop?: StageStatsDoc | null;
}

/** `caliper-learn profile --json` (profile.LatencyReport). */
export interface ProfileDoc {
  fps: number;
  ticks: number;
  budget_s: number;
  achievable_hz: number;
  jitter_s: number;
  frac_over_budget: number;
  overhead_s: number;
  stages: Record<string, StageStatsDoc>;
  chunk?: ChunkStatsDoc | null;
  findings: FindingDoc[];
}

/** `caliper-learn debug --json` (the CLI's `{policy_dir, findings}` payload). */
export interface DebugDoc {
  policy_dir: string;
  findings: FindingDoc[];
}

/** The dataset-doctor section of an autopsy (`caliper.data_doctor`). */
export interface DatasetSectionDoc {
  total_episodes: number;
  total_frames: number;
  fps: number;
  findings: FindingDoc[];
  root?: string;
  clean?: boolean;
}

/** `caliper-learn autopsy --json` (autopsy.AutopsyReport). */
export interface AutopsyDoc {
  policy_dir: string;
  dataset_root: string;
  verdict: string;
  dataset: DatasetSectionDoc;
  policy_findings: FindingDoc[];
  eval?: EvalDoc | null;
  latency?: ProfileDoc | null;
}

/** A recognized document. */
export type Detected =
  | { kind: "eval"; doc: EvalDoc }
  | { kind: "autopsy"; doc: AutopsyDoc }
  | { kind: "profile"; doc: ProfileDoc }
  | { kind: "debug"; doc: DebugDoc };

/** Detection outcome: a typed document, or why this file is not one. */
export type Detection = Detected | { kind: "unknown"; why: string };

/** A detected document plus where it was read from (the store's slice). */
export type LoadedVerdict = Detected & { path: string };

// ----- validation helpers (plain-English failures, additive-tolerant) -----

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function typeName(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "an array";
  switch (typeof v) {
    case "object":
      return "an object";
    case "string":
      return "a string";
    case "number":
      return Number.isFinite(v) ? "a number" : "a non-finite number";
    case "boolean":
      return "a boolean";
    case "undefined":
      return "missing";
    default:
      return typeof v;
  }
}

/** The first non-null complaint, or null when every check passed. */
function first(...checks: (string | null)[]): string | null {
  for (const c of checks) if (c !== null) return c;
  return null;
}

function numAt(o: Record<string, unknown>, key: string, at: string): string | null {
  const v = o[key];
  if (typeof v === "number" && Number.isFinite(v)) return null;
  return `${at}.${key} must be a number, got ${typeName(v)}`;
}

/** A number that the reporters legitimately write as null (or omit). */
function optNumAt(o: Record<string, unknown>, key: string, at: string): string | null {
  const v = o[key];
  if (v === undefined || v === null) return null;
  if (typeof v === "number" && Number.isFinite(v)) return null;
  return `${at}.${key} must be a number or null, got ${typeName(v)}`;
}

function strAt(o: Record<string, unknown>, key: string, at: string): string | null {
  return typeof o[key] === "string" ? null : `${at}.${key} must be a string, got ${typeName(o[key])}`;
}

function boolAt(o: Record<string, unknown>, key: string, at: string): string | null {
  return typeof o[key] === "boolean"
    ? null
    : `${at}.${key} must be true or false, got ${typeName(o[key])}`;
}

function objAt(o: Record<string, unknown>, key: string, at: string): string | null {
  return isObj(o[key]) ? null : `${at}.${key} must be an object, got ${typeName(o[key])}`;
}

function findingsAt(o: Record<string, unknown>, key: string, at: string): string | null {
  const v = o[key];
  if (!Array.isArray(v)) return `${at}.${key} must be a list of findings, got ${typeName(v)}`;
  for (let i = 0; i < v.length; i++) {
    const f = v[i];
    const where = `${at}.${key}[${i}]`;
    if (!isObj(f)) return `${where} must be a finding object, got ${typeName(f)}`;
    const bad = first(
      strAt(f, "code", where),
      strAt(f, "severity", where),
      strAt(f, "message", where),
    );
    if (bad !== null) return bad;
  }
  return null;
}

function stageStatsAt(v: unknown, at: string): string | null {
  if (!isObj(v)) return `${at} must be a stats object, got ${typeName(v)}`;
  return first(numAt(v, "p50", at), numAt(v, "p95", at), numAt(v, "p99", at), numAt(v, "max", at));
}

function validateEval(o: Record<string, unknown>, at: string): string | null {
  const bad = first(
    numAt(o, "n_episodes", at),
    numAt(o, "n_success", at),
    numAt(o, "success_rate", at),
    numAt(o, "ci95_low", at),
    numAt(o, "ci95_high", at),
    numAt(o, "mean_return", at),
    numAt(o, "median_return", at),
    optNumAt(o, "mean_steps_to_success", at),
    findingsAt(o, "findings", at),
  );
  if (bad !== null) return bad;
  const eps = o.episodes;
  if (!Array.isArray(eps)) return `${at}.episodes must be a list, got ${typeName(eps)}`;
  for (let i = 0; i < eps.length; i++) {
    const e = eps[i];
    const where = `${at}.episodes[${i}]`;
    if (!isObj(e)) return `${where} must be an episode object, got ${typeName(e)}`;
    const badEp = first(
      numAt(e, "seed", where),
      boolAt(e, "success", where),
      numAt(e, "steps", where),
      numAt(e, "episode_return", where),
      optNumAt(e, "final_distance", where),
    );
    if (badEp !== null) return badEp;
  }
  if (o.success_criterion !== undefined && o.success_criterion !== null) {
    const badCrit = strAt(o, "success_criterion", at);
    if (badCrit !== null) return badCrit;
  }
  return null;
}

function validateProfile(o: Record<string, unknown>, at: string): string | null {
  const bad = first(
    numAt(o, "fps", at),
    numAt(o, "ticks", at),
    numAt(o, "budget_s", at),
    numAt(o, "achievable_hz", at),
    numAt(o, "jitter_s", at),
    numAt(o, "frac_over_budget", at),
    numAt(o, "overhead_s", at),
    objAt(o, "stages", at),
    findingsAt(o, "findings", at),
  );
  if (bad !== null) return bad;
  const stages = o.stages as Record<string, unknown>;
  for (const name of Object.keys(stages)) {
    const badStage = stageStatsAt(stages[name], `${at}.stages.${name}`);
    if (badStage !== null) return badStage;
  }
  const chunk = o.chunk;
  if (chunk !== undefined && chunk !== null) {
    const where = `${at}.chunk`;
    if (!isObj(chunk)) return `${where} must be an object or null, got ${typeName(chunk)}`;
    const badChunk = first(
      strAt(chunk, "source", where),
      numAt(chunk, "n_refill_ticks", where),
      optNumAt(chunk, "period", where),
      stageStatsAt(chunk.refill, `${where}.refill`),
    );
    if (badChunk !== null) return badChunk;
    if (chunk.pop !== undefined && chunk.pop !== null) {
      const badPop = stageStatsAt(chunk.pop, `${where}.pop`);
      if (badPop !== null) return badPop;
    }
  }
  return null;
}

function validateDebug(o: Record<string, unknown>, at: string): string | null {
  return first(strAt(o, "policy_dir", at), findingsAt(o, "findings", at));
}

function validateAutopsy(o: Record<string, unknown>, at: string): string | null {
  const bad = first(
    strAt(o, "policy_dir", at),
    strAt(o, "dataset_root", at),
    strAt(o, "verdict", at),
    objAt(o, "dataset", at),
    findingsAt(o, "policy_findings", at),
  );
  if (bad !== null) return bad;
  const ds = o.dataset as Record<string, unknown>;
  const badDs = first(
    numAt(ds, "total_episodes", `${at}.dataset`),
    numAt(ds, "total_frames", `${at}.dataset`),
    numAt(ds, "fps", `${at}.dataset`),
    findingsAt(ds, "findings", `${at}.dataset`),
  );
  if (badDs !== null) return badDs;
  // the E and L sections are null when the autopsy ran without robot+task
  if (o.eval !== undefined && o.eval !== null) {
    if (!isObj(o.eval)) return `${at}.eval must be an object or null, got ${typeName(o.eval)}`;
    const badEval = validateEval(o.eval, `${at}.eval`);
    if (badEval !== null) return badEval;
  }
  if (o.latency !== undefined && o.latency !== null) {
    if (!isObj(o.latency))
      return `${at}.latency must be an object or null, got ${typeName(o.latency)}`;
    const badLat = validateProfile(o.latency, `${at}.latency`);
    if (badLat !== null) return badLat;
  }
  return null;
}

// ----- detection -----

const SWEEP_HINT =
  "this is a caliper-learn SWEEP ranking — a list of checkpoints, each with its own " +
  "eval result. Open one checkpoint's eval report instead; the verdict panel shows a single run.";

/** Recognize a parsed verdict document by its STRUCTURE (never its filename).
 *
 *  Fingerprints, checked most-specific first: `verdict` + `dataset` +
 *  `policy_findings` = the autopsy (it CONTAINS an eval and a latency report, so
 *  it must win over both); `success_rate` + `episodes` = eval; `achievable_hz` +
 *  `stages` = latency profile; `policy_dir` + `findings` = the policy debugger.
 *  A fingerprinted document is then type-checked, so a truncated or hand-edited
 *  file says what is wrong with it rather than rendering blanks. */
export function detectVerdict(json: unknown): Detection {
  if (Array.isArray(json)) {
    const looksSweep =
      json.length > 0 && isObj(json[0]) && "name" in json[0] && "result" in json[0];
    return {
      kind: "unknown",
      why: looksSweep
        ? SWEEP_HINT
        : "the file's top level is a JSON array; every caliper-learn verdict " +
          "(eval / autopsy / profile / debug) is a JSON object.",
    };
  }
  if (!isObj(json)) {
    return {
      kind: "unknown",
      why: `the file's top level is ${typeName(json)}; a caliper-learn verdict is a JSON object.`,
    };
  }
  if ("verdict" in json && "dataset" in json && "policy_findings" in json) {
    const why = validateAutopsy(json, "the autopsy report");
    return why !== null ? { kind: "unknown", why } : { kind: "autopsy", doc: json as unknown as AutopsyDoc };
  }
  if ("success_rate" in json && "episodes" in json) {
    const why = validateEval(json, "the eval report");
    return why !== null ? { kind: "unknown", why } : { kind: "eval", doc: json as unknown as EvalDoc };
  }
  if ("achievable_hz" in json && "stages" in json) {
    const why = validateProfile(json, "the latency profile");
    return why !== null ? { kind: "unknown", why } : { kind: "profile", doc: json as unknown as ProfileDoc };
  }
  if ("policy_dir" in json && "findings" in json) {
    const why = validateDebug(json, "the policy-debugger report");
    return why !== null ? { kind: "unknown", why } : { kind: "debug", doc: json as unknown as DebugDoc };
  }
  const keys = Object.keys(json);
  const found = keys.length === 0 ? "it has no keys at all" : `its keys are: ${keys.join(", ")}`;
  return {
    kind: "unknown",
    why:
      "no caliper-learn verdict fingerprint — expected `verdict`+`dataset` (autopsy), " +
      "`success_rate`+`episodes` (eval), `achievable_hz`+`stages` (profile) or " +
      `\`policy_dir\`+\`findings\` (debug), but ${found}.`,
  };
}

/** Panel title of each kind, with the doctor-style section letter. */
export function verdictLabel(kind: VerdictKind): string {
  switch (kind) {
    case "eval":
      return "Closed-loop eval (E)";
    case "autopsy":
      return "Autopsy (D · P · E · L)";
    case "profile":
      return "Latency profile (L)";
    case "debug":
      return "Policy debugger (P)";
  }
}

// ----- findings -----

/** Severity chip presentation. Unknown spellings keep their own text (a future
 *  reporter's severity must not silently read as one of ours) but take the warn
 *  colours — the conservative middle. */
export interface SevChip {
  cls: string;
  label: string;
}

export function sevChip(severity: string): SevChip {
  switch (severity) {
    case "error":
      return { cls: "sev-error", label: "ERR" };
    case "info":
      return { cls: "sev-info", label: "INFO" };
    case "warn":
    case "warning":
      return { cls: "sev-warn", label: "WARN" };
    default:
      return { cls: "sev-warn", label: severity.slice(0, 6).toUpperCase() || "?" };
  }
}

/** One finding, normalized across the four reporters (which anchor their
 *  findings on different fields) into a single render shape. */
export interface VerdictFinding {
  code: string;
  severity: string;
  message: string;
  /** The reporter's "what to do about it" line, when it has one. */
  suggestion: string | null;
  /** Machine anchors as short chips: "ep 3", "frame 21", "dof 1", the feature
   *  name, "value 0.0132" — in that order, absent ones skipped. */
  refs: string[];
}

function fmtRefNum(v: number): string {
  return String(Number(v.toPrecision(4)));
}

/** Normalize any reporter's findings list. Non-finding entries cannot appear —
 *  `detectVerdict` rejects a document whose findings are not findings. */
export function normalizeFindings(raw: FindingDoc[]): VerdictFinding[] {
  return raw.map((f) => {
    const refs: string[] = [];
    if (typeof f.episode === "number") refs.push(`ep ${f.episode}`);
    if (typeof f.frame === "number") refs.push(`frame ${f.frame}`);
    if (typeof f.dof === "number") refs.push(`dof ${f.dof}`);
    if (typeof f.feature === "string" && f.feature !== "") refs.push(f.feature);
    if (typeof f.value === "number" && Number.isFinite(f.value))
      refs.push(`value ${fmtRefNum(f.value)}`);
    return {
      code: f.code,
      severity: f.severity,
      message: f.message,
      suggestion: typeof f.fix_hint === "string" && f.fix_hint !== "" ? f.fix_hint : null,
      refs,
    };
  });
}

/** Severity tally of a findings list, for the `doctorSummary` one-liner. */
export function findingCounts(findings: { severity: string }[]): {
  errors: number;
  warnings: number;
  infos: number;
} {
  let errors = 0;
  let warnings = 0;
  let infos = 0;
  for (const f of findings) {
    if (f.severity === "error") errors += 1;
    else if (f.severity === "info") infos += 1;
    else warnings += 1;
  }
  return { errors, warnings, infos };
}

// ----- eval render prep -----

/** Geometry of the Wilson-interval bar, in PERCENT of the bar's width: the
 *  interval band [`lowPct`, `lowPct + widthPct`] and the point estimate marker.
 *
 *  Every value is clamped into [0, 100] — a 0/N run has `lowPct` 0 and an N/N
 *  run ends at 100, and neither may paint outside the track. A non-finite or
 *  inverted input collapses the band to zero width at the clamped low end
 *  rather than inventing a range. */
export interface WilsonBar {
  lowPct: number;
  highPct: number;
  widthPct: number;
  pointPct: number;
}

function clampUnit(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return Math.min(1, Math.max(0, x));
}

function pct(x: number): number {
  return Math.round(clampUnit(x) * 1e4) / 100;
}

export function wilsonBar(low: number, high: number, point: number): WilsonBar {
  const lowPct = pct(low);
  const highPct = Math.max(lowPct, pct(high));
  return {
    lowPct,
    highPct,
    widthPct: Math.round((highPct - lowPct) * 100) / 100,
    pointPct: pct(point),
  };
}

/** "3 / 6 episodes succeeded" — the headline count, never a bare rate. */
export function evalHeadline(doc: EvalDoc): string {
  const n = doc.n_episodes;
  return `${doc.n_success} / ${n} episode${n === 1 ? "" : "s"} succeeded`;
}

/** The interval sentence under the headline. */
export function evalInterval(doc: EvalDoc): string {
  return `${fmtPct(doc.success_rate)} · 95% CI [${fmtPct(doc.ci95_low)}, ${fmtPct(doc.ci95_high)}]`;
}

/** What the rate is a rate OF. An eval report always answers this: either the
 *  success predicate's own sentence, or the termination_fn that stood in for
 *  one (an older report that carries no criterion says so). */
export function successCriterion(doc: EvalDoc): string {
  if (typeof doc.success_criterion === "string" && doc.success_criterion !== "")
    return doc.success_criterion;
  if (doc.success_criterion === null) return "the task's termination_fn fired (no scene predicate)";
  return "not recorded by this report";
}

/** Percent with one decimal; "—" for a value that is not a number. */
export function fmtPct(x: number, digits = 1): string {
  return Number.isFinite(x) ? `${(x * 100).toFixed(digits)}%` : "—";
}

/** A number for a readout cell; "—" when absent. */
export function fmtNum(x: number | null | undefined, digits = 3): string {
  return typeof x === "number" && Number.isFinite(x) ? x.toFixed(digits) : "—";
}

/** Seconds → milliseconds text; "—" when absent. */
export function fmtMs(seconds: number | null | undefined, digits = 3): string {
  return typeof seconds === "number" && Number.isFinite(seconds)
    ? (seconds * 1e3).toFixed(digits)
    : "—";
}

// ----- profile render prep -----

/** Canonical stage order (the profiler's own): the tick is built in this order
 *  and `total` closes it. Stages a future profiler adds are appended, sorted,
 *  so an unknown stage still shows up. */
const STAGE_ORDER = ["obs_build", "inference", "step", "total"];

/** One row of the percentile table, in milliseconds. `overBudget` means this
 *  stage's own p95 already exceeds the per-tick deadline. */
export interface LatencyRow {
  stage: string;
  p50Ms: string;
  p95Ms: string;
  p99Ms: string;
  maxMs: string;
  overBudget: boolean;
}

function stageRow(stage: string, s: StageStatsDoc, budgetS: number): LatencyRow {
  return {
    stage,
    p50Ms: fmtMs(s.p50),
    p95Ms: fmtMs(s.p95),
    p99Ms: fmtMs(s.p99),
    maxMs: fmtMs(s.max),
    overBudget: Number.isFinite(budgetS) && s.p95 > budgetS,
  };
}

/** Percentile rows of a latency profile, canonical stages first. */
export function latencyRows(doc: ProfileDoc): LatencyRow[] {
  const names = Object.keys(doc.stages);
  const known = STAGE_ORDER.filter((s) => names.includes(s));
  const extra = names.filter((s) => !STAGE_ORDER.includes(s)).sort();
  return [...known, ...extra].map((s) => stageRow(s, doc.stages[s], doc.budget_s));
}

/** The chunked-policy rows (refill / pop), or null for an unchunked profile.
 *  `pop` is absent when every measured tick was a refill. */
export function chunkRows(doc: ProfileDoc): LatencyRow[] | null {
  const c = doc.chunk;
  if (!c) return null;
  const rows = [stageRow("refill", c.refill, doc.budget_s)];
  if (c.pop) rows.push(stageRow("pop", c.pop, doc.budget_s));
  return rows;
}

/** "config, every 4 ticks · 10 refills" — how the refill ticks were identified. */
export function chunkSummary(c: ChunkStatsDoc): string {
  const period = typeof c.period === "number" ? `every ${c.period} ticks` : "period undetermined";
  return `${c.source}, ${period} · ${c.n_refill_ticks} refill ticks`;
}

/** Key/value readouts above a latency table. */
export function latencyMetrics(doc: ProfileDoc): [string, string][] {
  return [
    ["rate", `${doc.fps} Hz`],
    ["budget", `${fmtMs(doc.budget_s)} ms`],
    ["achievable", `${doc.achievable_hz.toFixed(0)} Hz`],
    ["ticks", String(doc.ticks)],
    ["jitter", `${fmtMs(doc.jitter_s)} ms`],
    ["over budget", fmtPct(doc.frac_over_budget, 0)],
  ];
}

/** Key/value readouts of an eval report (everything not in the headline). */
export function evalMetrics(doc: EvalDoc): [string, string][] {
  return [
    ["mean return", fmtNum(doc.mean_return)],
    ["median return", fmtNum(doc.median_return)],
    ["steps to success", fmtNum(doc.mean_steps_to_success, 1)],
    ["episodes", String(doc.n_episodes)],
  ];
}

/** Key/value readouts of an autopsy's dataset section. */
export function datasetMetrics(ds: DatasetSectionDoc): [string, string][] {
  return [
    ["episodes", String(ds.total_episodes)],
    ["frames", String(ds.total_frames)],
    ["fps", String(ds.fps)],
  ];
}
