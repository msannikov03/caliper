import { useStore } from "../store";

export function Hud() {
  const robot = useStore((s) => s.robot);
  const frames = useStore((s) => s.frames);
  const ikOk = useStore((s) => s.ikOk);
  const res = useStore((s) => s.ikResidual);
  const error = useStore((s) => s.error);

  if (error) return <div className="hud err">error: {error}</div>;
  if (!robot) return <div className="hud">loading robot…</div>;

  const tip = frames[robot.tip];
  const pos = tip ? [tip[12], tip[13], tip[14]] : [0, 0, 0];
  const tipName = robot.frames[robot.tip]?.name ?? "—";

  return (
    <div className="hud">
      <div>
        tip <b>{tipName}</b>
      </div>
      <div className="mono">
        xyz {pos.map((p) => p.toFixed(3)).join("  ")}
      </div>
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
