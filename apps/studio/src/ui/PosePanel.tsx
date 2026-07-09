import { useState } from "react";
import { useStore } from "../store";
import "./panels.css";

/** "62.1%" for a utilization fraction; "∞" survives a degenerate limit. */
function pct(x: number): string {
  return Number.isFinite(x) ? `${(x * 100).toFixed(1)}%` : "∞";
}

/** Compact cycle-time / conditioning / utilization readout for the last plan
 *  (rides on TrajectoryDto.report; sim rollouts and graph clips carry none). */
function PlanReport() {
  const robot = useStore((s) => s.robot);
  const report = useStore((s) => s.traj?.report ?? null);
  if (!robot || !report) return null;
  const jointName = (i: number) => (i >= 0 ? robot.jointNames[i] : "—");
  return (
    <div className="report">
      <div className="pose-title">Report</div>
      <div className="rp-row">
        <span>cycle time</span>
        <b>{report.cycleTime.toFixed(3)} s</b>
      </div>
      <div className="rp-row">
        <span>min σ_min</span>
        <b>{report.minSigmaMin.toExponential(2)}</b>
      </div>
      <div className="rp-row" title={`worst joint velocity vs limit: ${jointName(report.velUtilJoint)}`}>
        <span>vel util</span>
        <b className={report.velUtil > 1 ? "bad" : ""}>
          {pct(report.velUtil)} · {jointName(report.velUtilJoint)}
        </b>
      </div>
      <div className="rp-row" title={`worst joint acceleration vs limit: ${jointName(report.accUtilJoint)}`}>
        <span>acc util</span>
        <b className={report.accUtil > 1 ? "bad" : ""}>
          {pct(report.accUtil)} · {jointName(report.accUtilJoint)}
        </b>
      </div>
      {report.limitMargin !== null && (
        <div className="rp-row" title={`tightest distance to a position limit: ${jointName(report.limitMarginJoint)}`}>
          <span>limit margin</span>
          <b className={report.limitMargin < 0 ? "bad" : ""}>
            {report.limitMargin.toFixed(3)} · {jointName(report.limitMarginJoint)}
          </b>
        </div>
      )}
    </div>
  );
}

export function PosePanel() {
  const robot = useStore((s) => s.robot);
  const mode = useStore((s) => s.mode);
  const poses = useStore((s) => s.poses);
  const savePose = useStore((s) => s.savePose);
  const deletePose = useStore((s) => s.deletePose);
  const planMoveToPose = useStore((s) => s.planMoveToPose);
  const [name, setName] = useState("");
  if (!robot) return null;
  return (
    <aside className="pose-panel">
      <div className="pose-title">Poses</div>
      <div className="pose-add">
        <input value={name} placeholder="name" onChange={(e) => setName(e.target.value)} />
        <button
          disabled={!name}
          onClick={() => {
            void savePose(name);
            setName("");
          }}
        >
          Save current
        </button>
      </div>
      <ul>
        {poses.map((p) => (
          <li key={p.name}>
            <span>{p.name}</span>
            <button
              disabled={mode === "simulate"}
              onClick={() => void planMoveToPose(p.name)}
            >
              Move to
            </button>
            <button onClick={() => void deletePose(p.name)}>×</button>
          </li>
        ))}
      </ul>
      <PlanReport />
    </aside>
  );
}
