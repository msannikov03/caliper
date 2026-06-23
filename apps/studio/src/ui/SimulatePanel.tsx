import { useStore } from "../store";

export function SimulatePanel() {
  const mode = useStore((s) => s.mode);
  const robot = useStore((s) => s.robot);
  const simGravity = useStore((s) => s.simGravity);
  const simDamping = useStore((s) => s.simDamping);
  const simTraj = useStore((s) => s.simTraj);
  const run = useStore((s) => s.runGravityDrop);
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
