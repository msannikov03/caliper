// ============================================================
// DataMode.tsx — the "Data" Studio mode: a LeRobotDataset browser
// and offline episode editor. Left: dataset panel (open/refresh,
// info header, episode table). Right: episode detail (per-channel
// uPlot series, tag chips, delete/split/merge edit bar with an
// INLINE confirm — never a browser dialog).
//
// Deliberately robot-independent: everything here talks to the
// dataset_* backend commands and touches none of the robot state,
// so Data mode is reachable with no URDF loaded. All pure logic
// lives in ./episodes.ts (vitest-covered); this file renders.
// ============================================================

import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { useStore } from "../store";
import type { DatasetEpisodeSeries, DatasetSummary } from "../store";
import { EpisodeChart } from "./EpisodeChart";
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
} from "./episodes";
import "./data.css";

/** Native FOLDER picker → open the dataset + switch to Data mode. Module-scope
 *  (mirrors Toolbar.openUrdf) so the ⌘K palette shares this one implementation. */
export async function openDatasetDialog(): Promise<void> {
  const picked = await open({ multiple: false, directory: true, title: "Open dataset folder" });
  if (typeof picked !== "string") return; // dialog cancelled
  const st = useStore.getState();
  if (st.mode !== "data") st.setMode("data");
  await useStore.getState().openDataset(picked);
}

function DatasetInfo({ ds }: { ds: DatasetSummary }) {
  const rows: [string, string][] = [
    ["robot", ds.robotType ?? "—"],
    ["fps", String(ds.fps)],
    ["version", ds.codebaseVersion],
    ["episodes", String(ds.totalEpisodes)],
    ["frames", String(ds.totalFrames)],
    ["tasks", String(ds.totalTasks)],
  ];
  return (
    <div className="data-info">
      {rows.map(([k, v]) => (
        <div className="di-metric" key={k}>
          <span className="di-key">{k}</span>
          <span className="di-val">{v}</span>
        </div>
      ))}
      <div className="di-path" title={ds.path}>
        {ds.path}
      </div>
    </div>
  );
}

function EpisodeTable({
  ds,
  sel,
  onSelect,
}: {
  ds: DatasetSummary;
  sel: number | null;
  onSelect: (i: number) => void;
}) {
  return (
    <div className="data-table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>#</th>
            <th>len</th>
            <th>dur</th>
            <th>tasks</th>
            <th>tags</th>
          </tr>
        </thead>
        <tbody>
          {ds.episodes.map((r) => {
            const v = rowView(r);
            return (
              <tr
                key={r.index}
                className={sel === r.index ? "sel" : ""}
                onClick={() => onSelect(r.index)}
              >
                <td className="num">{v.index}</td>
                <td className="num">{v.length}</td>
                <td className="num">{v.duration}</td>
                <td className="tasks" title={v.tasks}>
                  {v.tasks || "—"}
                </td>
                <td className="tags">
                  {v.tags.map((t) => (
                    <span className="tag-chip sm" key={t}>
                      {t}
                    </span>
                  ))}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function EpisodeDetail({
  ds,
  sel,
  series,
  loading,
}: {
  ds: DatasetSummary;
  sel: number;
  series: DatasetEpisodeSeries | null;
  loading: boolean;
}) {
  const row = ds.episodes[sel];
  const [splitFrame, setSplitFrame] = useState(0);
  const [tagDraft, setTagDraft] = useState("");
  // inline destructive-op confirm (small strip in the edit bar area)
  const [confirm, setConfirm] = useState<{ label: string; run: () => void } | null>(null);
  const len = row?.length ?? 0;

  // a new selection (or a re-listed episode) resets the local edit state
  useEffect(() => {
    setSplitFrame(Math.floor(len / 2));
    setTagDraft("");
    setConfirm(null);
  }, [sel, len]);

  if (!row) return null;

  const partner = mergePartner(sel, ds.episodes.length);
  const split = clampSplitFrame(splitFrame, row.length);
  const live = series && series.episode === sel ? series : null;
  const cursorT =
    live && split !== null
      ? (live.times[cursorIndex(split, live.stride, live.times.length)] ?? null)
      : null;

  const commitTags = (tags: string[]) => {
    if (tags !== row.tags) void useStore.getState().setDatasetTags(sel, tags);
  };

  return (
    <div className="data-detail-body">
      <div className="data-editbar">
        <span className="eyebrow accent">Episode {sel}</span>
        <span className="de-meta">
          {row.length} frames · {fmtDuration(row.durationS)}
        </span>
        <span className="de-spacer" />
        <label className="de-split">
          <span>split @</span>
          <input
            type="number"
            min={1}
            max={Math.max(1, row.length - 1)}
            value={splitFrame}
            onChange={(e) => setSplitFrame(Number(e.target.value))}
          />
        </label>
        <button
          className="btn ghost"
          disabled={loading || split === null}
          title={split === null ? "episode too short to split" : ""}
          onClick={() =>
            split !== null &&
            setConfirm({
              label: `Split episode ${sel} at frame ${split}?`,
              run: () => void useStore.getState().splitDatasetEpisode(sel, split),
            })
          }
        >
          Split
        </button>
        <button
          className="btn ghost"
          disabled={loading || partner === null}
          title={partner === null ? "no adjacent episode" : ""}
          onClick={() =>
            partner !== null &&
            setConfirm({
              label: `Merge episodes ${sel} + ${partner}?`,
              run: () => void useStore.getState().mergeDatasetEpisodes(sel, partner),
            })
          }
        >
          Merge with next
        </button>
        <button
          className="btn ghost de-danger"
          disabled={loading}
          onClick={() =>
            setConfirm({
              label: `Delete episode ${sel}? This rewrites the dataset on disk.`,
              run: () => void useStore.getState().deleteDatasetEpisodes([sel]),
            })
          }
        >
          Delete
        </button>
      </div>

      {/* draggable split cursor — mirrored as the dashed line on every chart */}
      <input
        className="de-split-range"
        type="range"
        min={1}
        max={Math.max(1, row.length - 1)}
        value={split ?? 1}
        disabled={row.length < 2}
        onChange={(e) => setSplitFrame(Number(e.target.value))}
      />

      {confirm && (
        <div className="data-confirm">
          <span>{confirm.label}</span>
          <button
            className="btn de-danger"
            onClick={() => {
              confirm.run();
              setConfirm(null);
            }}
          >
            Confirm
          </button>
          <button className="btn ghost" onClick={() => setConfirm(null)}>
            Cancel
          </button>
        </div>
      )}

      <div className="data-tags">
        <span className="eyebrow">Tags</span>
        <div className="tag-edit">
          {row.tags.map((t) => (
            <span className="tag-chip" key={t}>
              {t}
              <button
                aria-label={`remove tag ${t}`}
                onClick={() => commitTags(removeTag(row.tags, t))}
              >
                ×
              </button>
            </span>
          ))}
          <input
            value={tagDraft}
            placeholder="add tag ⏎"
            spellCheck={false}
            onChange={(e) => setTagDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key !== "Enter") return;
              e.preventDefault();
              commitTags(addTag(row.tags, tagDraft));
              setTagDraft("");
            }}
          />
        </div>
      </div>

      <div className="data-charts">
        {!live && <div className="data-empty">loading series…</div>}
        {live &&
          live.channels.map((ch) => {
            const feat = ds.features.find((f) => f.name === ch.name);
            const aligned = alignChannel(live.times, ch);
            const dims = aligned.slice(1);
            const labels = dimLabels(ch.name, ch.series.length, feat?.names ?? null).slice(
              0,
              dims.length,
            );
            if (dims.length === 0) return null;
            return (
              <div className="data-chart-card" key={ch.name}>
                <div className="dc-head">
                  <span className="eyebrow">{ch.name}</span>
                  <span className="dc-meta">
                    {dims.length}d · stride {live.stride}
                  </span>
                </div>
                <EpisodeChart times={live.times} series={dims} labels={labels} cursorT={cursorT} />
                <div className="dc-legend">
                  {labels.map((l, i) => (
                    <span className="dc-key" key={l}>
                      <i style={{ background: seriesColor(i) }} />
                      {l}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
      </div>
    </div>
  );
}

export function DataMode() {
  const ds = useStore((s) => s.dataset);
  const loading = useStore((s) => s.datasetLoading);
  const error = useStore((s) => s.datasetError);
  const sel = useStore((s) => s.datasetEpisode);
  const series = useStore((s) => s.datasetSeries);
  const selectEpisode = useStore((s) => s.selectDatasetEpisode);
  const refreshDataset = useStore((s) => s.refreshDataset);

  return (
    <div className="data-root">
      <aside className="data-side">
        <div className="data-actions">
          <span className="eyebrow">Dataset</span>
          <span className="de-spacer" />
          <button className="btn ghost" disabled={loading} onClick={() => void openDatasetDialog()}>
            Open dataset…
          </button>
          <button
            className="btn ghost"
            disabled={!ds || loading}
            onClick={() => void refreshDataset()}
          >
            Refresh
          </button>
        </div>
        {error && <div className="data-banner">{error}</div>}
        {ds ? (
          <>
            <DatasetInfo ds={ds} />
            <EpisodeTable ds={ds} sel={sel} onSelect={(i) => void selectEpisode(i)} />
          </>
        ) : (
          !error && (
            <div className="data-empty">
              Open a LeRobotDataset v3.0 folder to browse, plot and edit its episodes. Works with
              no robot loaded.
            </div>
          )
        )}
      </aside>
      <section className="data-detail">
        {ds && sel !== null ? (
          <EpisodeDetail ds={ds} sel={sel} series={series} loading={loading} />
        ) : (
          <div className="data-empty">{ds ? "Select an episode to plot and edit it." : ""}</div>
        )}
      </section>
    </div>
  );
}
