// ============================================================
// doctor.ts — pure, headless logic for the two doctor surfaces
// (asset doctor in the error banner, dataset doctor in Data
// mode): wire types for `urdf_doctor` / `dataset_doctor`, the
// severity→chip mapping, the compact-list capping rule, the
// repair gate, and the finding→episode-row jump. No React, no
// Tauri, no store — the vitest suite drives everything here
// directly (Hud.tsx / DataMode.tsx are just renderers).
// ============================================================

/** Asset-doctor severities as the backend serializes them. */
export type AssetSeverity = "error" | "warn" | "info";
/** Dataset-doctor severities (the engine spells the middle one out). */
export type DataSeverity = "error" | "warning" | "info";

/** One `urdf_doctor` finding (stable codes A001…A014). */
export interface DoctorFinding {
  code: string;
  severity: AssetSeverity;
  message: string;
  fixHint: string | null;
  autoFixable: boolean;
}

/** `urdf_doctor` result. `repair`/`after` only present when repair ran. */
export interface DoctorReport {
  findings: DoctorFinding[];
  errors: number;
  warnings: number;
  infos: number;
  repair: { out: string; applied: string[]; skipped: string[] } | null;
  after: DoctorFinding[] | null;
}

/** One `dataset_doctor` finding (stable codes D001…D015). */
export interface DataDoctorFinding {
  code: string;
  severity: DataSeverity;
  feature: string | null;
  episode: number | null;
  dof: number | null;
  message: string;
  fixHint: string;
}

/** `dataset_doctor` result: header facts + findings, most-severe first. */
export interface DataDoctorReport {
  totalEpisodes: number;
  totalFrames: number;
  fps: number;
  errors: number;
  warnings: number;
  infos: number;
  findings: DataDoctorFinding[];
}

/** CSS modifier class of a severity chip (both doctors share the chips). */
export function sevClass(sev: AssetSeverity | DataSeverity): string {
  return sev === "error" ? "sev-error" : sev === "info" ? "sev-info" : "sev-warn";
}

/** Chip label — short, fixed-width-friendly. */
export function sevLabel(sev: AssetSeverity | DataSeverity): string {
  return sev === "error" ? "ERR" : sev === "info" ? "INFO" : "WARN";
}

/** Cap a findings list for a compact banner/panel. The list arrives sorted
 *  most-severe-first from the engine, so a plain prefix keeps the worst;
 *  `hidden` feeds the "+N more" line. A non-positive cap shows everything. */
export function visibleFindings<T>(findings: T[], cap: number): { shown: T[]; hidden: number } {
  if (cap <= 0 || findings.length <= cap) return { shown: findings.slice(), hidden: 0 };
  return { shown: findings.slice(0, cap), hidden: findings.length - cap };
}

/** True when a repair run would actually change something — gates the
 *  "Repair & reload" button. */
export function canRepair(report: DoctorReport | null): boolean {
  return report !== null && report.findings.some((f) => f.autoFixable);
}

/** One-line severity tally, zero counts omitted: "2 errors · 1 warning".
 *  Empty report → "no findings". */
export function doctorSummary(errors: number, warnings: number, infos: number): string {
  const parts: string[] = [];
  const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;
  if (errors > 0) parts.push(plural(errors, "error"));
  if (warnings > 0) parts.push(plural(warnings, "warning"));
  if (infos > 0) parts.push(plural(infos, "info"));
  return parts.length ? parts.join(" · ") : "no findings";
}

/** The episode-table row a finding jumps to when clicked: its episode ref,
 *  when present AND a valid row index (episode indices are contiguous 0-based
 *  after every caliper edit/rewrite); otherwise null (not clickable). */
export function findingEpisode(f: DataDoctorFinding, episodeCount: number): number | null {
  if (f.episode === null || !Number.isInteger(f.episode)) return null;
  return f.episode >= 0 && f.episode < episodeCount ? f.episode : null;
}

/** Machine refs of a dataset finding as short chips: "ep 3", "dof 1", and the
 *  feature name — in that order, empties skipped. */
export function findingRefs(f: DataDoctorFinding): string[] {
  const refs: string[] = [];
  if (f.episode !== null) refs.push(`ep ${f.episode}`);
  if (f.dof !== null) refs.push(`dof ${f.dof}`);
  if (f.feature !== null && f.feature !== "") refs.push(f.feature);
  return refs;
}
