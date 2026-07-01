import { useStore } from "../store";
import "./panels.css";

export function SimulatePanel() {
  const mode = useStore((s) => s.mode);
  const robot = useStore((s) => s.robot);
  const simGravity = useStore((s) => s.simGravity);
  const simDamping = useStore((s) => s.simDamping);
  const simTraj = useStore((s) => s.simTraj);
  const run = useStore((s) => s.runGravityDrop);
  const runControl = useStore((s) => s.runControl);
  const runPlan = useStore((s) => s.runPlan);
  const checkCollision = useStore((s) => s.checkCollision);
  const collision = useStore((s) => s.collision);
  if (mode !== "simulate" || !robot) return null;
  const noInertia = !robot.hasInertia;
  const driftPct = simTraj ? (simTraj.energyDrift * 100).toFixed(3) : null;
  const energyOk = simTraj ? simTraj.energyDrift < 1e-3 : false;
  return (
    <aside className="sim-panel">
      <h3>Simulate</h3>
      <button
        disabled={noInertia}
        title={noInertia ? "robot has no inertial data" : ""}
        onClick={() => void run()}
      >
        ⤓ Gravity drop
      </button>
      <button
        disabled={noInertia}
        title={noInertia ? "robot has no inertial data" : "computed-torque control back to home"}
        onClick={() => void runControl(new Array(robot.ndof).fill(0))}
      >
        ⌖ Drive to home
      </button>
      <button
        title="collision-free RRT plan back to home"
        onClick={() => void runPlan(new Array(robot.ndof).fill(0))}
      >
        ⛬ Plan to home
      </button>
      <button onClick={() => void checkCollision(null)}>⚠ Check collision</button>
      {collision && (
        <div className="sim-badges">
          <span className={collision.collision ? "badge bad" : "badge ok"}>
            {collision.collision ? "COLLISION" : "clear"}
          </span>
          {collision.selfPairs.map(([a, b], i) => (
            <span className="badge bad" key={i}>
              {a} ↔ {b}
            </span>
          ))}
          {collision.worldHits.map((f, i) => (
            <span className="badge bad" key={`w${i}`}>
              {f} · world
            </span>
          ))}
        </div>
      )}
      <label>
        <input
          type="checkbox"
          checked={simGravity}
          onChange={(e) => useStore.setState({ simGravity: e.target.checked })}
        />{" "}
        gravity
      </label>
      <label>
        damping {simDamping.toFixed(2)}
        <input
          type="range"
          min={0}
          max={2}
          step={0.05}
          value={simDamping}
          onChange={(e) => useStore.setState({ simDamping: parseFloat(e.target.value) })}
        />
      </label>
      {simTraj && (
        <div className="sim-badges">
          <span className={energyOk ? "badge ok" : "badge"}>
            energy {energyOk ? "✓" : ""} drift {driftPct}%
          </span>
          {simTraj.settled && <span className="badge ok">settled</span>}
        </div>
      )}
      {noInertia && <p className="hint">load showcase6 or dyn_pendulum2 (they carry &lt;inertial&gt;)</p>}
    </aside>
  );
}
