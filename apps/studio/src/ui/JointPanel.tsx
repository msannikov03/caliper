import { useStore } from "../store";

const PI = Math.PI;

export function JointPanel() {
  const robot = useStore((s) => s.robot);
  const q = useStore((s) => s.q);
  const setJoint = useStore((s) => s.setJoint);
  const playing = useStore((s) => s.playing);
  const mode = useStore((s) => s.mode);
  if (!robot) return null;
  const locked = playing || mode === "simulate"; // sim/playback own the pose

  return (
    <aside className="joint-panel">
      <h3>
        {robot.name} · {robot.ndof} DOF
      </h3>
      {robot.jointNames.map((name, i) => {
        const kind = robot.jointKinds[i];
        const lim = robot.limits[i];
        const [lo, hi] = lim ?? (kind === "prismatic" ? [-0.5, 0.5] : [-PI, PI]);
        const v = q[i] ?? 0;
        return (
          <div className="joint-row" key={i}>
            <label title={name}>
              <span className="glyph">{kind === "revolute" ? "⟳" : "↔"}</span>
              {name}
            </label>
            <input
              type="range"
              min={lo}
              max={hi}
              step={0.001}
              value={v}
              disabled={locked}
              onChange={(e) => setJoint(i, parseFloat(e.target.value))}
            />
            <span className="val">{v.toFixed(3)}</span>
          </div>
        );
      })}
    </aside>
  );
}
