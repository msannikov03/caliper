import { useState } from "react";
import { useStore } from "../store";
import "./panels.css";

const PI = Math.PI;

/** signed, tabular-friendly formatter using a true unicode minus (−) */
function sgn(x: number, decimals: number): string {
  const s = x < 0 ? "−" : "+";
  return s + Math.abs(x).toFixed(decimals);
}
const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

export function JointPanel() {
  const robot = useStore((s) => s.robot);
  const q = useStore((s) => s.q);
  const setJoint = useStore((s) => s.setJoint);
  const playing = useStore((s) => s.playing);
  const mode = useStore((s) => s.mode);
  const [active, setActive] = useState<number | null>(null);
  if (!robot) return null;
  const locked = playing || mode === "simulate"; // sim/playback own the pose

  return (
    <aside className="joint-panel">
      <h3>Joints</h3>
      <div className="jp-sub">
        {robot.name} · {robot.ndof} DOF
      </div>
      {robot.jointNames.map((name, i) => {
        const kind = robot.jointKinds[i];
        const lim = robot.limits[i];
        const [lo, hi] = lim ?? (kind === "prismatic" ? [-0.5, 0.5] : [-PI, PI]);
        const v = q[i] ?? 0;
        const unit = kind === "prismatic" ? "m" : "rad";
        const span = hi - lo || 1;
        const pct = clamp01((v - lo) / span) * 100;
        const centerPct = clamp01((0 - lo) / span) * 100;
        return (
          <div className={active === i ? "joint active" : "joint"} key={i}>
            <div className="j-top">
              <div className="j-name">
                <span className="j-idx">{i + 1}</span>
                <span className="j-label" title={name}>
                  {name}
                </span>
              </div>
              <div className="j-read">
                {sgn(v, 4)}
                <span className="u"> {unit}</span>
              </div>
            </div>
            <div className="j-track">
              <div className="j-center-tick" style={{ left: `${centerPct}%` }} />
              <div className="j-fill" style={{ width: `${pct}%` }} />
              <div className="j-knob" style={{ left: `${pct}%` }} />
              <input
                type="range"
                min={lo}
                max={hi}
                step={0.001}
                value={v}
                disabled={locked}
                aria-label={name}
                onFocus={() => setActive(i)}
                onBlur={() => setActive((a) => (a === i ? null : a))}
                onPointerDown={() => setActive(i)}
                onChange={(e) => setJoint(i, parseFloat(e.target.value))}
              />
            </div>
            <div className="j-lims">
              <span>{sgn(lo, 3)}</span>
              <span>{sgn(hi, 3)}</span>
            </div>
          </div>
        );
      })}
    </aside>
  );
}
