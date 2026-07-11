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

// ---- camera thumbnails ----

/** Thumbnails per strip (the backend clamps to the episode length). */
export const THUMB_COUNT = 8;

/** Evenly-spaced full-resolution frame indices for a thumbnail strip,
 *  endpoints included — MUST stay in lockstep with `thumb_picks` in
 *  `src-tauri/src/lib.rs`, which selects the frames the backend actually
 *  encodes; this mirror maps a clicked thumb back to its source frame. */
export function thumbFrameIndices(length: number, count: number): number[] {
  if (!Number.isFinite(length) || length <= 0) return [];
  const n = Math.min(Math.max(Math.floor(count), 1), Math.floor(length));
  if (n === 1) return [0];
  return Array.from({ length: n }, (_, i) => Math.floor((i * (length - 1)) / (n - 1)));
}

/** Decode `dataset_episode_thumbs`' length-prefixed binary framing — u32 LE
 *  image count, then per image a u32 LE byte length + the encoded bytes —
 *  into one byte view per image. Throws on malformed framing (truncation or
 *  trailing bytes) instead of yielding broken images. Pure bytes → bytes so
 *  the vitest suite can assert exact round-trips without Blob support. */
export function decodeThumbFrames(buf: ArrayBuffer): Uint8Array[] {
  const view = new DataView(buf);
  if (buf.byteLength < 4) throw new Error("thumb framing: truncated header");
  const count = view.getUint32(0, true);
  const out: Uint8Array[] = [];
  let off = 4;
  for (let k = 0; k < count; k++) {
    if (off + 4 > buf.byteLength) throw new Error(`thumb framing: truncated length of image ${k}`);
    const len = view.getUint32(off, true);
    off += 4;
    if (off + len > buf.byteLength) throw new Error(`thumb framing: truncated bytes of image ${k}`);
    out.push(new Uint8Array(buf.slice(off, off + len)));
    off += len;
  }
  if (off !== buf.byteLength) throw new Error("thumb framing: trailing bytes");
  return out;
}

/** [`decodeThumbFrames`] wrapped into displayable Blobs (feed each through
 *  `URL.createObjectURL` — and revoke the URLs when done). */
export function decodeThumbs(buf: ArrayBuffer, mime = "image/png"): Blob[] {
  return decodeThumbFrames(buf).map((bytes) => new Blob([bytes], { type: mime }));
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
