import { useStore } from "../store";
import { baseName } from "../commands";
import { canRepair, sevClass, sevLabel, visibleFindings } from "../doctor/doctor";
import "./hud.css";

/** Compact banner cap: the worst findings (engine pre-sorts) + a "+N more". */
const HUD_FINDINGS_CAP = 6;

/** Asset-doctor readout inside the error banner: severity-chipped findings
 *  plus "Repair & reload" when a mechanical fix exists. Pure renderer — the
 *  capping/gating logic lives in doctor.ts (vitest-covered). */
function DoctorFindings() {
  const doctor = useStore((s) => s.doctor);
  const repairing = useStore((s) => s.repairing);
  const repairAndReload = useStore((s) => s.repairAndReload);
  if (!doctor) return null;

  const { shown, hidden } = visibleFindings(doctor.findings, HUD_FINDINGS_CAP);
  return (
    <div className="doctor-findings">
      {shown.map((f, i) => (
        <div className="df-row" key={`${f.code}-${i}`} title={f.fixHint ?? undefined}>
          <span className={`sev-chip ${sevClass(f.severity)}`}>{sevLabel(f.severity)}</span>
          <span className="df-code">{f.code}</span>
          <span className="df-msg">{f.message}</span>
        </div>
      ))}
      {hidden > 0 && <div className="df-more">+{hidden} more (see `caliper doctor`)</div>}
      {canRepair(doctor) && (
        <button
          className="btn df-repair"
          disabled={repairing}
          onClick={() => void repairAndReload()}
        >
          {repairing ? "Repairing…" : "Repair & reload"}
        </button>
      )}
    </div>
  );
}

export function Hud() {
  const robot = useStore((s) => s.robot);
  const frames = useStore((s) => s.frames);
  const ikOk = useStore((s) => s.ikOk);
  const res = useStore((s) => s.ikResidual);
  const error = useStore((s) => s.error);
  const doctor = useStore((s) => s.doctor);
  const repairedFrom = useStore((s) => s.repairedFrom);

  // the doctor can have Error findings even when the robot LOADED (e.g. a
  // dropped collider) — both cases share this one banner
  if (error || doctor)
    return (
      <div className="hud err">
        <span className="eyebrow">{error ? "Error" : "Asset doctor"}</span>
        {error && <div className="mono">{error}</div>}
        <DoctorFindings />
      </div>
    );
  if (!robot)
    return (
      <div className="hud">
        <span className="eyebrow">Loading robot…</span>
      </div>
    );

  const tip = frames[robot.tip];
  const pos = tip ? [tip[12], tip[13], tip[14]] : [0, 0, 0];
  const tipName = robot.frames[robot.tip]?.name ?? "—";

  return (
    <div className="hud">
      <span className="eyebrow">Tool center point</span>
      <div className="tip-name">{tipName}</div>
      <div className="mono">xyz {pos.map((p) => p.toFixed(3)).join("  ")}</div>
      {repairedFrom && (
        <div className="mono repaired-note" title={repairedFrom}>
          repaired copy of {baseName(repairedFrom)}
        </div>
      )}
      <div className={`ik ${ikOk ? "ok" : ikOk === false ? "fail" : ""}`}>
        {ikOk == null
          ? "drag the tip gizmo to solve IK"
          : ikOk
            ? `IK ✓  r=${res?.toExponential(1)}`
            : `IK ✗  r=${res?.toExponential(1)}`}
      </div>
    </div>
  );
}
