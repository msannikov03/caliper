// ============================================================
// episodes.ts — pure helpers for Data mode (the dataset browser).
// Headless: no React, no Tauri, no store — the vitest suite drives
// every function here directly. DataMode.tsx/EpisodeChart.tsx are
// renderers over these.
// ============================================================

import type { DatasetChannel, DatasetEpisodeRow } from "../store";

/** Per-dim series labels for one channel: the dataset's element names when
 *  present (and dimension-consistent), else `name[i]`. */
export function dimLabels(name: string, dim: number, names: string[] | null): string[] {
  if (names && names.length === dim) return names;
  return Array.from({ length: dim }, (_, i) => `${name}[${i}]`);
}

/** uPlot aligned-data shape for one channel: [times, dim0, dim1, …].
 *  Dims whose length disagrees with `times` are dropped (a malformed episode
 *  must degrade to a shorter plot, never crash uPlot). */
export function alignChannel(times: number[], ch: DatasetChannel): number[][] {
  return [times, ...ch.series.filter((s) => s.length === times.length)];
}

/** Compact mono duration readout: "8.4s" under a minute, "1:23.4" above. */
export function fmtDuration(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "—";
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

/** Episode-table row view: everything the table renders, pre-formatted. */
export interface EpisodeRowView {
  index: number;
  length: number;
  duration: string;
  tasks: string;
  tags: string[];
}
export function rowView(r: DatasetEpisodeRow): EpisodeRowView {
  return {
    index: r.index,
    length: r.length,
    duration: fmtDuration(r.durationS),
    tasks: r.tasks.join(" · "),
    tags: r.tags,
  };
}

/** The episode `index` can merge with, or null: merge is adjacent-only, so the
 *  only candidate is the NEXT row (when it exists). */
export function mergePartner(index: number | null, episodeCount: number): number | null {
  if (index === null || index < 0) return null;
  return index + 1 < episodeCount ? index + 1 : null;
}

/** Clamp a split point into the valid range [1, length-1] (both halves must
 *  keep at least one frame). Null when the episode is too short to split or
 *  the input is not a finite number. */
export function clampSplitFrame(frame: number, length: number): number | null {
  if (!Number.isFinite(frame) || length < 2) return null;
  return Math.min(Math.max(Math.round(frame), 1), length - 1);
}

/** Map a FULL-resolution frame index onto the decimated sample index (the
 *  series arrives with `stride` applied), clamped into [0, nSamples-1]. */
export function cursorIndex(frame: number, stride: number, nSamples: number): number {
  if (nSamples <= 0) return 0;
  const i = Math.round(frame / Math.max(1, stride));
  return Math.min(Math.max(i, 0), nSamples - 1);
}

/** Chip-input add: trimmed, non-empty, de-duplicated; otherwise unchanged
 *  (same array identity, so callers can skip the backend write). */
export function addTag(tags: string[], raw: string): string[] {
  const t = raw.trim();
  if (t === "" || tags.includes(t)) return tags;
  return [...tags, t];
}

/** Chip-input remove (unchanged identity when the tag is absent). */
export function removeTag(tags: string[], tag: string): string[] {
  return tags.includes(tag) ? tags.filter((x) => x !== tag) : tags;
}

/** Categorical stroke palette for multi-dim episode plots (accent first). */
export const SERIES_COLORS = [
  "#7c82ff",
  "#3dd68c",
  "#ffb02e",
  "#ff5c6c",
  "#56c8d8",
  "#c792ea",
  "#e0af68",
  "#8a9099",
] as const;
export function seriesColor(i: number): string {
  return SERIES_COLORS[((i % SERIES_COLORS.length) + SERIES_COLORS.length) % SERIES_COLORS.length];
}
