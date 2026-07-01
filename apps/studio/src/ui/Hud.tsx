import { useStore } from "../store";
import "./hud.css";

export function Hud() {
  const robot = useStore((s) => s.robot);
  const frames = useStore((s) => s.frames);
  const ikOk = useStore((s) => s.ikOk);
  const res = useStore((s) => s.ikResidual);
  const error = useStore((s) => s.error);

  if (error)
    return (
      <div className="hud err">
        <span className="eyebrow">Error</span>
        <div className="mono">{error}</div>
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
