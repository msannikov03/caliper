import { useStore } from "../store";
import { frameIndexAt, hasContactEngine, MAX_PROPS } from "../sim/props";
import "./panels.css";

/** Kind glyphs for the compact prop list rows. */
const KIND_GLYPH: Record<string, string> = { box: "▢", sphere: "◯", cylinder: "◫" };

export function SimulatePanel() {
  const mode = useStore((s) => s.mode);
  const robot = useStore((s) => s.robot);
  const simGravity = useStore((s) => s.simGravity);
  const simDamping = useStore((s) => s.simDamping);
  const simTraj = useStore((s) => s.simTraj);
  const playhead = useStore((s) => s.playhead);
  const run = useStore((s) => s.runGravityDrop);
  const runControl = useStore((s) => s.runControl);
  const runPlan = useStore((s) => s.runPlan);
  const checkCollision = useStore((s) => s.checkCollision);
  const collision = useStore((s) => s.collision);
  const simEngines = useStore((s) => s.simEngines);
  const simEngine = useStore((s) => s.simEngine);
  const simProps = useStore((s) => s.simProps);
  const setSimEngine = useStore((s) => s.setSimEngine);
  const addSimProp = useStore((s) => s.addSimProp);
  const removeSimProp = useStore((s) => s.removeSimProp);
  const runContactSim = useStore((s) => s.runContactSim);
  const live = useStore((s) => s.live);
  const startLive = useStore((s) => s.startLive);
  const stopLive = useStore((s) => s.stopLive);
  const pauseLive = useStore((s) => s.pauseLive);
  const resetLive = useStore((s) => s.resetLive);
  if (mode !== "simulate" || !robot) return null;
  const noInertia = !robot.hasInertia;
  const driftPct = simTraj ? (simTraj.energyDrift * 100).toFixed(3) : null;
  const energyOk = simTraj ? simTraj.energyDrift < 1e-3 : false;
  // without mujoco in the build, the panel renders EXACTLY as before: no
  // toggle, no contact UI (pinned by the palette gating tests on the same flag)
  const mujoco = hasContactEngine(simEngines);
  const contactView = mujoco && simEngine === "mujoco";
  // live contact readout: the count at the CURRENT playback instant, read from
  // the per-frame array with the same rounding the robot pose playback uses
  const contactClip = simTraj?.kind === "contact" && simTraj.contacts?.length ? simTraj : null;
  const ncon = contactClip
    ? contactClip.contacts![frameIndexAt(playhead, contactClip.dt, contactClip.contacts!.length)]
    : null;
  // a live session streams the pose in; baking a clip over it would fight the
  // stream, so every rollout button goes inert until it is stopped
  const liveEngine = mujoco ? "mujoco" : "builtin";
  const bakeOff = noInertia || live !== null;
  const bakeTitle = (ready: string) =>
    live ? "stop live to bake" : noInertia ? "robot has no inertial data" : ready;
  return (
    <aside className="sim-panel">
      <h3>Simulate</h3>
      {mujoco && (
        <div className="segmented sim-engine">
          <button
            className={contactView ? "" : "active"}
            onClick={() => setSimEngine("builtin")}
          >
            Builtin
          </button>
          <button className={contactView ? "active" : ""} onClick={() => setSimEngine("mujoco")}>
            Contact
          </button>
        </div>
      )}
      <div className="live-block">
        <div className="sim-badges">
          {live && <span className="badge live">● LIVE</span>}
          <span className="badge" title="engine backing a live session">
            {live ? live.engine : liveEngine}
          </span>
          {live && <span className="badge">t {live.t.toFixed(1)}s</span>}
          {live && live.engine === "mujoco" && (
            <span className={live.ncon > 0 ? "badge bad" : "badge ok"}>
              {live.ncon > 0 ? `CONTACT ×${live.ncon}` : "no contact"}
            </span>
          )}
        </div>
        {live ? (
          <div className="live-controls">
            <button
              title={live.paused ? "resume integration" : "freeze integration"}
              onClick={() => void pauseLive(!live.paused)}
            >
              {live.paused ? "▶ Resume" : "⏸ Pause"}
            </button>
            <button title="restart at the starting pose" onClick={() => void resetLive()}>
              ↺ Reset
            </button>
            <button title="end the session" onClick={() => void stopLive()}>
              ■ Stop
            </button>
          </div>
        ) : (
          <button
            disabled={noInertia}
            title={noInertia ? "robot has no inertial data" : "stream physics live from this pose"}
            onClick={() => void startLive()}
          >
            ◉ Start live
          </button>
        )}
      </div>
      {contactView ? (
        <>
          <div className="prop-add">
            <button
              disabled={simProps.length >= MAX_PROPS}
              title={simProps.length >= MAX_PROPS ? `max ${MAX_PROPS} props` : "add a free box"}
              onClick={() => addSimProp("box")}
            >
              + box
            </button>
            <button
              disabled={simProps.length >= MAX_PROPS}
              title={simProps.length >= MAX_PROPS ? `max ${MAX_PROPS} props` : "add a free sphere"}
              onClick={() => addSimProp("sphere")}
            >
              + sphere
            </button>
          </div>
          {simProps.length > 0 && (
            <ul className="prop-list">
              {simProps.map((p, i) => (
                <li key={p.name}>
                  <span className="p-kind">{KIND_GLYPH[p.kind] ?? "◇"}</span>
                  <span className="p-name">{p.name}</span>
                  <button className="p-del" title="remove prop" onClick={() => removeSimProp(i)}>
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
          <button
            disabled={bakeOff}
            title={bakeTitle("passive drop with contacts")}
            onClick={() => void runContactSim("drop")}
          >
            ⤓ Gravity drop
          </button>
          <button
            disabled={bakeOff}
            title={bakeTitle("computed-torque hold at this pose")}
            onClick={() => void runContactSim("hold")}
          >
            ⊙ Hold pose
          </button>
          <button
            disabled={bakeOff}
            title={bakeTitle("computed-torque drive to home")}
            onClick={() => void runContactSim("drive_to", new Array(robot.ndof).fill(0))}
          >
            ⌖ Drive to home
          </button>
          {ncon !== null && (
            <div className="sim-badges">
              <span className={ncon > 0 ? "badge bad" : "badge ok"}>
                {ncon > 0 ? `CONTACT ×${ncon}` : "no contact"}
              </span>
              {contactClip!.settled && <span className="badge ok">settled</span>}
              {contactClip!.lint?.length === 0 && (
                <span className="badge ok" title="contact stability lint C001–C003: no findings">
                  stability ✓
                </span>
              )}
            </div>
          )}
          {simTraj?.kind === "contact" && simTraj.lint && simTraj.lint.length > 0 && (
            <ul className="lint-list">
              {simTraj.lint.map((f) => (
                <li key={f.code} title={f.message}>
                  <span className="badge bad">{f.code}</span>
                  <span className="lint-fix">{f.suggestion}</span>
                </li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <>
          <button disabled={bakeOff} title={bakeTitle("")} onClick={() => void run()}>
            ⤓ Gravity drop
          </button>
          <button
            disabled={bakeOff}
            title={bakeTitle("computed-torque control back to home")}
            onClick={() => void runControl(new Array(robot.ndof).fill(0))}
          >
            ⌖ Drive to home
          </button>
          <button
            disabled={live !== null}
            title={live ? "stop live to bake" : "collision-free RRT plan back to home"}
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
          {simTraj && simTraj.kind !== "contact" && (
            <div className="sim-badges">
              <span className={energyOk ? "badge ok" : "badge"}>
                energy {energyOk ? "✓" : ""} drift {driftPct}%
              </span>
              {simTraj.settled && <span className="badge ok">settled</span>}
            </div>
          )}
        </>
      )}
      {noInertia && (
        <p className="hint">load showcase6 or dyn_pendulum2 (they carry &lt;inertial&gt;)</p>
      )}
    </aside>
  );
}
