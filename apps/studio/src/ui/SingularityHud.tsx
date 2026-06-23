import { useStore } from "../store";
import { severity, rampColor } from "../sevColor";

export function SingularityHud() {
  const r = useStore((s) => s.report);
  if (!r) return null;
  const f = severity(r.sigmaMin, r.epsActivate);
  const kindColor =
    r.kind === "none" ? "#3ad29f" : r.kind === "boundary" ? "#e5484d" : "#e0a83a";
  const kappa = r.conditionNumber == null ? "∞" : r.conditionNumber.toExponential(2);
  return (
    <div className="hud sing">
      <div className="badge" style={{ background: kindColor }}>
        {r.kind === "none" ? "REGULAR" : r.kind.toUpperCase()}
      </div>
      <div className="mono">manip ∏σ {r.manipulability.toExponential(2)}</div>
      <div className="mono">cond κ {kappa}</div>
      <div className="mono">σmin {r.sigmaMin.toExponential(2)}</div>
      <div className="bar">
        <i style={{ width: `${f * 100}%`, background: rampColor(f) }} />
      </div>
      <div className="mono dim">σ {r.sigma.map((s) => s.toExponential(1)).join("  ")}</div>
    </div>
  );
}
