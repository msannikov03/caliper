import { useStore } from "../store";
import { severity } from "../sevColor";
import "./hud.css";

// Direction A · Instrument palette (matches App.css tokens; three.js/SVG can't
// resolve CSS vars for the continuous ellipsoid tint, so map to token hex here).
function rampToken(f: number): string {
  if (f >= 0.66) return "var(--good)";
  if (f >= 0.33) return "var(--warn)";
  return "var(--singular)";
}

export function SingularityHud() {
  const r = useStore((s) => s.report);
  if (!r) return null;

  const f = severity(r.sigmaMin, r.epsActivate); // 1 = well-conditioned, 0 = singular
  const c = rampToken(f); // continuous condition tint for the ellipse + w readout

  // Status LED / label follow the singularity KIND (preserves original semantics).
  const status =
    r.kind === "none"
      ? { cls: "led-good", label: "NOMINAL", color: "var(--good)" }
      : r.kind === "boundary"
        ? { cls: "led-singular", label: "BOUNDARY", color: "var(--singular)" }
        : { cls: "led-warn", label: r.kind.toUpperCase(), color: "var(--warn)" };

  const kappa = r.conditionNumber == null ? "∞" : r.conditionNumber.toExponential(2);
  const smax = Math.max(r.sigma[0], r.sigma[1], r.sigma[2]);

  // 2-D manipulability ellipse: rx ∝ σ_max, ry ∝ σ_min → flattens toward the
  // singular set. Axis-aligned so the coloured principal axes read cleanly.
  const RMAX = 44;
  const rx = RMAX;
  const ry = smax > 0 ? Math.max(2, (r.sigmaMin / smax) * RMAX) : 2;

  return (
    <div className="sing-hud hud-card">
      <div className="h-head">
        <span className="h-title">Singularity · Manip</span>
        <span className="h-status" style={{ color: status.color }}>
          <span className={`led ${status.cls}`} />
          {status.label}
        </span>
      </div>
      <div className="h-body">
        <div className="ellipse-wrap">
          <div className="ellipse-box">
            <svg viewBox="0 0 104 104" width="104" height="104" aria-hidden="true">
              {/* reference rings */}
              <circle cx="52" cy="52" r="40" fill="none" stroke="rgba(255,255,255,0.05)" />
              <circle cx="52" cy="52" r="26" fill="none" stroke="rgba(255,255,255,0.04)" />
              <circle cx="52" cy="52" r="13" fill="none" stroke="rgba(255,255,255,0.035)" />
              {/* the manipulability ellipse — colour degrades good→warn→singular */}
              <ellipse
                cx="52"
                cy="52"
                rx={rx}
                ry={ry}
                fill={c}
                fillOpacity={0.14}
                stroke={c}
                strokeOpacity={0.78}
                strokeWidth={1.2}
              />
              {/* principal axes = singular values (robotics-true axis triad) */}
              <line x1={52 - rx} y1="52" x2={52 + rx} y2="52" stroke="var(--ax-x)" strokeWidth="1.2" strokeOpacity="0.85" />
              <line x1="52" y1={52 - ry} x2="52" y2={52 + ry} stroke="var(--ax-y)" strokeWidth="1.2" strokeOpacity="0.7" />
              <circle cx={52 + rx} cy="52" r="2.2" fill="var(--ax-x)" />
              <circle cx="52" cy={52 + ry} r="2.2" fill="var(--ax-y)" />
              <circle cx="52" cy="52" r="1.8" fill="var(--text)" />
            </svg>
          </div>
          <div className="ell-metrics">
            <div className="metric">
              <span className="m-key">w = √det(JJᵀ)</span>
              <span className="m-val" style={{ color: c }}>
                {r.manipulability.toExponential(2)}
              </span>
            </div>
            <div className="metric">
              <span className="m-key">cond κ</span>
              <span className="m-val">{kappa}</span>
            </div>
            <div className="metric">
              <span className="m-key">σ_min</span>
              <span className="m-val">{r.sigmaMin.toExponential(2)}</span>
            </div>
            <div className="metric">
              <span className="m-key">σ_max</span>
              <span className="m-val">{smax.toExponential(2)}</span>
            </div>
          </div>
        </div>
        <div className="gauge">
          <div className="track-mask" />
          <div className="needle" style={{ left: `${f * 100}%` }} />
        </div>
        <div className="gauge-legend">
          <span>SINGULAR</span>
          <span>PROXIMITY</span>
          <span>WELL-COND</span>
        </div>
      </div>
    </div>
  );
}
