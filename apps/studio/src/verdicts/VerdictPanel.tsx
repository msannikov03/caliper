// ============================================================
// VerdictPanel.tsx — read-only viewers for the JSON a learning
// run leaves behind (`caliper-learn eval|autopsy|profile|debug
// --json`). Rendered at the top of Data mode's detail column, so
// a run's verdict sits next to the dataset it was trained on.
//
// Renderers only: detection, validation and every number/string
// this file prints come from ./verdicts.ts (vitest-covered). The
// severity chips and finding rows reuse the doctor panel's
// vocabulary — one visual language for "here is what is wrong".
// ============================================================

import { open } from "@tauri-apps/plugin-dialog";
import { useStore } from "../store";
import { doctorSummary } from "../doctor/doctor";
import type {
  AutopsyDoc,
  DatasetSectionDoc,
  DebugDoc,
  EvalDoc,
  FindingDoc,
  LatencyRow,
  LoadedVerdict,
  ProfileDoc,
  VerdictFinding,
} from "./verdicts";
import {
  chunkRows,
  chunkSummary,
  datasetMetrics,
  evalHeadline,
  evalInterval,
  evalMetrics,
  findingCounts,
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

/** Native FILE picker → read + detect a verdict JSON. Module-scope (mirrors
 *  `openDatasetDialog`) so any caller shares this one implementation. */
export async function openVerdictDialog(): Promise<void> {
  const picked = await open({
    multiple: false,
    directory: false,
    title: "Open caliper-learn verdict",
    filters: [{ name: "caliper-learn verdict (*.json)", extensions: ["json"] }],
  });
  if (typeof picked !== "string") return; // dialog cancelled
  await useStore.getState().openVerdict(picked);
}

function Metrics({ rows }: { rows: [string, string][] }) {
  return (
    <div className="vc-metrics">
      {rows.map(([k, v]) => (
        <div className="di-metric" key={k}>
          <span className="di-key">{k}</span>
          <span className="di-val">{v}</span>
        </div>
      ))}
    </div>
  );
}

/** Findings list, doctor vocabulary: severity chip, stable code, message, the
 *  machine anchors as chips, and the reporter's fix hint under it. */
function Findings({ raw, clean }: { raw: FindingDoc[]; clean: string }) {
  const findings: VerdictFinding[] = normalizeFindings(raw);
  const { errors, warnings, infos } = findingCounts(findings);
  if (findings.length === 0) return <div className="data-empty vc-clean">{clean}</div>;
  return (
    <div className="vc-findings">
      <div className="vc-tally">{doctorSummary(errors, warnings, infos)}</div>
      {findings.map((f, i) => {
        const chip = sevChip(f.severity);
        return (
          <div className="vc-finding" key={`${f.code}-${i}`}>
            <div className="vc-frow">
              <span className={`sev-chip ${chip.cls}`}>{chip.label}</span>
              <span className="dd-code">{f.code}</span>
              <span className="dd-msg">{f.message}</span>
              <span className="dd-refs">
                {f.refs.map((r) => (
                  <span className="dd-ref" key={r}>
                    {r}
                  </span>
                ))}
              </span>
            </div>
            {f.suggestion && <div className="vc-fix">fix: {f.suggestion}</div>}
          </div>
        );
      })}
    </div>
  );
}

/** The Wilson-95 interval as a track: the band is the interval the run actually
 *  supports, the marker the point estimate. A 6-episode run paints a band half
 *  the track wide — which is the whole point of showing it. */
function WilsonBar({ doc }: { doc: EvalDoc }) {
  const bar = wilsonBar(doc.ci95_low, doc.ci95_high, doc.success_rate);
  return (
    <div className="vc-wilson">
      <div className="vw-track">
        <div
          className="vw-band"
          style={{ left: `${bar.lowPct}%`, width: `${bar.widthPct}%` }}
          title={`95% CI [${fmtPct(doc.ci95_low)}, ${fmtPct(doc.ci95_high)}]`}
        />
        <div
          className="vw-point"
          style={{ left: `${bar.pointPct}%` }}
          title={`success rate ${fmtPct(doc.success_rate)}`}
        />
      </div>
      <div className="vw-scale">
        <span>0%</span>
        <span>{evalInterval(doc)}</span>
        <span>100%</span>
      </div>
    </div>
  );
}

function EpisodeRows({ doc }: { doc: EvalDoc }) {
  if (doc.episodes.length === 0)
    return <div className="data-empty">no per-episode rows in this report.</div>;
  return (
    <div className="vc-table-wrap">
      <table className="data-table vc-table">
        <thead>
          <tr>
            <th>seed</th>
            <th>result</th>
            <th>steps</th>
            <th>return</th>
            <th>final dist</th>
          </tr>
        </thead>
        <tbody>
          {doc.episodes.map((e, i) => (
            <tr key={`${e.seed}-${i}`}>
              <td className="num">{e.seed}</td>
              <td className={e.success ? "vc-ok" : "vc-fail"}>{e.success ? "pass" : "fail"}</td>
              <td className="num">{e.steps}</td>
              <td className="num">{fmtNum(e.episode_return)}</td>
              <td className="num">{fmtNum(e.final_distance)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** E-section: the headline count, the interval it is worth, WHAT success meant,
 *  the aggregate readouts, every seed, then the findings. */
function EvalBody({ doc }: { doc: EvalDoc }) {
  return (
    <div className="vc-section">
      <div className="vc-headline">
        <span className="vc-big">{evalHeadline(doc)}</span>
      </div>
      <WilsonBar doc={doc} />
      <div className="vc-criterion">
        <span className="di-key">success =</span> {successCriterion(doc)}
      </div>
      <Metrics rows={evalMetrics(doc)} />
      <EpisodeRows doc={doc} />
      <Findings raw={doc.findings} clean="No eval findings — the run itself looks trustworthy." />
    </div>
  );
}

function LatencyTable({ rows, caption }: { rows: LatencyRow[]; caption: string }) {
  return (
    <div className="vc-table-wrap">
      <table className="data-table vc-table">
        <thead>
          <tr>
            <th>{caption}</th>
            <th>p50 ms</th>
            <th>p95 ms</th>
            <th>p99 ms</th>
            <th>max ms</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.stage} className={r.overBudget ? "vc-over" : ""}>
              <td>{r.stage}</td>
              <td className="num">{r.p50Ms}</td>
              <td className="num">{r.p95Ms}</td>
              <td className="num">{r.p99Ms}</td>
              <td className="num">{r.maxMs}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** L-section: the rate it can actually hold, the per-stage percentiles, and —
 *  for a chunked policy — the refill tick that is the real deadline. */
function ProfileBody({ doc }: { doc: ProfileDoc }) {
  const chunk = chunkRows(doc);
  return (
    <div className="vc-section">
      <Metrics rows={latencyMetrics(doc)} />
      <LatencyTable rows={latencyRows(doc)} caption="stage" />
      {chunk && doc.chunk && (
        <>
          <div className="vc-sub">
            <span className="eyebrow">Chunked policy</span>
            <span className="de-meta">{chunkSummary(doc.chunk)}</span>
          </div>
          <LatencyTable rows={chunk} caption="inference" />
        </>
      )}
      <Findings
        raw={doc.findings}
        clean={`No latency findings — the loop holds ${doc.fps} Hz with headroom.`}
      />
    </div>
  );
}

/** P-section. */
function DebugBody({ doc }: { doc: DebugDoc }) {
  return (
    <div className="vc-section">
      <div className="vc-path" title={doc.policy_dir}>
        {doc.policy_dir}
      </div>
      <Findings raw={doc.findings} clean="No policy findings — every P-check passed." />
    </div>
  );
}

function DatasetBody({ ds }: { ds: DatasetSectionDoc }) {
  return (
    <div className="vc-section">
      <Metrics rows={datasetMetrics(ds)} />
      <Findings raw={ds.findings} clean="No dataset findings — this data looks trainable." />
    </div>
  );
}

function NotRun({ what }: { what: string }) {
  return <div className="data-empty">not run — {what}</div>;
}

/** The autopsy: the one-paragraph verdict first (it is the whole point), then
 *  the four sections it was assembled from, each labelled with its letter. */
function AutopsyBody({ doc }: { doc: AutopsyDoc }) {
  return (
    <div className="vc-section">
      <div className="vc-verdict">{doc.verdict}</div>
      <div className="vc-pair">
        <span className="di-key">policy</span>
        <span className="vc-path" title={doc.policy_dir}>
          {doc.policy_dir}
        </span>
      </div>
      <div className="vc-pair">
        <span className="di-key">dataset</span>
        <span className="vc-path" title={doc.dataset_root}>
          {doc.dataset_root}
        </span>
      </div>

      <div className="vc-band">
        <span className="eyebrow accent">D</span> dataset doctor
      </div>
      <DatasetBody ds={doc.dataset} />

      <div className="vc-band">
        <span className="eyebrow accent">P</span> policy debugger
      </div>
      <div className="vc-section">
        <Findings raw={doc.policy_findings} clean="No policy findings — every P-check passed." />
      </div>

      <div className="vc-band">
        <span className="eyebrow accent">E</span> closed-loop eval
      </div>
      {doc.eval ? (
        <EvalBody doc={doc.eval} />
      ) : (
        <NotRun what="the autopsy had no robot + task to roll out." />
      )}

      <div className="vc-band">
        <span className="eyebrow accent">L</span> deploy latency
      </div>
      {doc.latency ? (
        <ProfileBody doc={doc.latency} />
      ) : (
        <NotRun what="the autopsy had no robot + task to profile the control loop with." />
      )}
    </div>
  );
}

/** One-line summary in the panel header: the fact a reader wants before
 *  scrolling. */
function headerSummary(v: LoadedVerdict): string {
  switch (v.kind) {
    case "eval":
      return `${evalHeadline(v.doc)} · ${evalInterval(v.doc)}`;
    case "profile":
      return `${v.doc.achievable_hz.toFixed(0)} Hz achievable at a ${v.doc.fps} Hz budget · ${fmtPct(
        v.doc.frac_over_budget,
        0,
      )} of ticks over`;
    case "debug":
      return `${v.doc.findings.length} finding${v.doc.findings.length === 1 ? "" : "s"}`;
    case "autopsy":
      return `${v.doc.dataset.total_episodes} episodes · ${v.doc.policy_findings.length} policy finding${
        v.doc.policy_findings.length === 1 ? "" : "s"
      }`;
  }
}

/** The verdict card: nothing when no verdict is open and no open failed. */
export function VerdictPanel() {
  const verdict = useStore((s) => s.verdict);
  const error = useStore((s) => s.verdictError);
  const clear = useStore((s) => s.clearVerdict);

  if (!verdict) {
    if (!error) return null;
    return (
      <div className="verdict-card">
        <div className="vc-head">
          <span className="eyebrow accent">Verdict</span>
          <button className="dd-close" aria-label="dismiss verdict error" onClick={clear}>
            ×
          </button>
        </div>
        <div className="data-banner vc-banner">{error}</div>
      </div>
    );
  }

  return (
    <div className="verdict-card">
      <div className="vc-head">
        <span className="eyebrow accent">{verdictLabel(verdict.kind)}</span>
        <span className="vc-sum">{headerSummary(verdict)}</span>
        <button className="dd-close" aria-label="close verdict" onClick={clear}>
          ×
        </button>
      </div>
      <div className="vc-file" title={verdict.path}>
        {verdict.path}
      </div>
      {verdict.kind === "eval" && <EvalBody doc={verdict.doc} />}
      {verdict.kind === "profile" && <ProfileBody doc={verdict.doc} />}
      {verdict.kind === "debug" && <DebugBody doc={verdict.doc} />}
      {verdict.kind === "autopsy" && <AutopsyBody doc={verdict.doc} />}
    </div>
  );
}
