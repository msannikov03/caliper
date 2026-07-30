import { REC_FPS_CHOICES, useStore } from "../store";
import { frameIndexAt, hasContactEngine, MAX_PROPS } from "../sim/props";
import { gripperControl, gripperSeated } from "../sim/live";
import { recFpsChoices, successBadge } from "../sim/task";
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
  const liveDriving = useStore((s) => s.liveDriving);
  const startLive = useStore((s) => s.startLive);
  const stopLive = useStore((s) => s.stopLive);
  const pauseLive = useStore((s) => s.pauseLive);
  const resetLive = useStore((s) => s.resetLive);
  const toggleGripper = useStore((s) => s.toggleGripper);
  const rec = useStore((s) => s.liveRec);
  const recTask = useStore((s) => s.liveRecTask);
  const recFps = useStore((s) => s.liveRecFps);
  const recHint = useStore((s) => s.liveRecHint);
  const recDone = useStore((s) => s.liveRecDone);
  const recordStart = useStore((s) => s.recordStart);
  const recordStop = useStore((s) => s.recordStop);
  const recordFinish = useStore((s) => s.recordFinish);
  const openRecordedDataset = useStore((s) => s.openRecordedDataset);
  const task = useStore((s) => s.task);
  if (mode !== "simulate" || !robot) return null;
  const noInertia = !robot.hasInertia;
  const driftPct = simTraj ? (simTraj.energyDrift * 100).toFixed(3) : null;
  const energyOk = simTraj ? simTraj.energyDrift < 1e-3 : false;
  // the engine segmented control is the ONE engine choice on this panel: it
  // picks the contact-bake view AND the engine a live session starts on. Without
  // mujoco in the build the Contact half is inert, so the contact UI below still
  // never appears (the same flag the palette gating tests pin).
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
  const liveEngine = contactView ? "mujoco" : "builtin";
  // the gripper control reflects the COMMAND intent; the jaw readout appears
  // only when the measurement disagrees with it — a jaw stopped short is
  // either still travelling or holding something, and saying which would be
  // a guess the sim cannot make
  const grip = gripperControl(live?.gripper ?? null, live?.gripperState ?? null);
  const gripState = live?.gripperState ?? null;
  const jaw =
    live?.gripper && gripState && !gripperSeated(live.gripper, gripState) ? gripState : null;
  const jawTitle =
    jaw && live?.gripper
      ? `commanded ${jaw.closed ? "closed" : "open"} — the jaw is at ${jaw.q.toFixed(3)}, ` +
        `target ${(jaw.closed ? live.gripper.closedTarget : live.gripper.openTarget).toFixed(3)}`
      : "";
  // the streamed verdict of the task's success predicate — per instant, never
  // latched, and absent entirely when nothing is scoring the session
  const verdict = successBadge(live?.success, task?.successDescription ?? null);
  const bakeOff = noInertia || live !== null;
  const bakeTitle = (ready: string) =>
    live ? "stop live to bake" : noInertia ? "robot has no inertial data" : ready;
  return (
    <aside className="sim-panel">
      <h3>Simulate</h3>
      <div className="segmented sim-engine">
        <button
          className={contactView ? "" : "active"}
          disabled={live !== null}
          title={live ? "stop the live session to switch engine" : "rigid-body engine"}
          onClick={() => setSimEngine("builtin")}
        >
          Builtin
        </button>
        <button
          className={contactView ? "active" : ""}
          disabled={!mujoco || live !== null}
          title={
            !mujoco
              ? "this build has no mujoco contact engine"
              : live
                ? "stop the live session to switch engine"
                : "MuJoCo contacts + free props"
          }
          onClick={() => setSimEngine("mujoco")}
        >
          Contact
        </button>
      </div>
      <div className="live-block">
        {/* the loaded task, so the panel never looks like a hand-built scene */}
        {task && (
          <div className="sim-badges">
            <span className="badge" title={task.path}>
              task {task.name}
            </span>
            {task.horizonS !== null && <span className="badge">horizon {task.horizonS}s</span>}
          </div>
        )}
        {task?.successDescription && <p className="hint">{task.successDescription}</p>}
        {/* a task IS its scene and its verdict, and both are contact-engine
            features — a build without one would otherwise just look empty */}
        {task && !mujoco && (
          <p className="hint rec-hint">
            this build has no mujoco contact engine — the task's props and verdict are inert
          </p>
        )}
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
          {live && liveDriving && (
            <span className="badge live" title="an input is moving the hold target">
              ⇢ driving
            </span>
          )}
          {rec?.recording && (
            <span className="badge rec" title={`recording "${rec.task}" at ${rec.fps} fps`}>
              ● REC
            </span>
          )}
          {verdict && (
            <span className={verdict.className} title={verdict.title}>
              {verdict.label}
            </span>
          )}
        </div>
        {live ? (
          <>
            <div className="live-controls">
              <button
                title={live.paused ? "resume integration (space)" : "freeze integration (space)"}
                onClick={() => void pauseLive(!live.paused)}
              >
                {live.paused ? "▶ Resume" : "⏸ Pause"}
              </button>
              <button
                title={
                  rec?.recording
                    ? "restart at the starting pose — DISCARDS the current take"
                    : "restart at the starting pose"
                }
                onClick={() => void resetLive()}
              >
                ↺ Reset
              </button>
              <button title="end the session" onClick={() => void stopLive()}>
                ■ Stop
              </button>
            </div>
            {/* the gripper channel the session auto-detected; the control is
                present either way so a robot without one says WHY */}
            <div className="grip-row">
              <button
                className={grip.closed ? "grip closed" : "grip"}
                disabled={grip.disabled}
                title={grip.title}
                onClick={() => void toggleGripper()}
              >
                {grip.label}
              </button>
              {jaw && (
                <span className="badge" title={jawTitle}>
                  jaw {jaw.q.toFixed(3)}
                </span>
              )}
              {live.held && (
                <span
                  className="badge held"
                  title={`${live.held} is welded while the gripper stays closed (heuristic)`}
                >
                  HELD {live.held}
                </span>
              )}
            </div>
            <p className="hint">
              space freeze · G grip · sliders/gizmo drive · [ ] pick joint, −/= jog · gamepad:
              sticks=tip, A=pause, B=reset, X=grip
            </p>
            <div className="rec-row">
              <input
                value={recTask}
                disabled={rec?.recording}
                placeholder="task label"
                title="what this demonstration does — each episode carries its own"
                onChange={(e) => useStore.setState({ liveRecTask: e.target.value })}
              />
              <select
                value={recFps}
                disabled={rec !== null}
                title={
                  rec
                    ? "the open dataset's rate — finish it to record at another"
                    : "recorded frames per second"
                }
                onChange={(e) => useStore.setState({ liveRecFps: Number(e.target.value) })}
              >
                {recFpsChoices(REC_FPS_CHOICES, task?.fps).map((f) => (
                  <option key={f} value={f}>
                    {f} fps
                  </option>
                ))}
              </select>
            </div>
            {rec?.recording ? (
              <div className="rec-take">
                <button
                  className="rec-stop"
                  title="keep this take as an episode"
                  onClick={() => void recordStop(true)}
                >
                  ■ stop take ({rec.frames} frames)
                </button>
                <button
                  className="rec-drop"
                  title="throw this take away"
                  onClick={() => void recordStop(false)}
                >
                  discard
                </button>
              </div>
            ) : (
              <button
                className="rec-arm"
                disabled={!recTask.trim()}
                title={
                  recTask.trim()
                    ? rec
                      ? `record another episode into ${rec.root}`
                      : "pick a dataset directory and record an episode"
                    : "give the episode a task label first"
                }
                onClick={() => void recordStart(recTask, recFps)}
              >
                ● record
              </button>
            )}
            {rec && rec.episodesSaved > 0 && (
              <div className="rec-take">
                <span className="badge" title={rec.root}>
                  episodes: {rec.episodesSaved}
                </span>
                <button
                  disabled={rec.recording}
                  title={
                    rec.recording ? "stop the take first" : "close the dataset (writes its meta/)"
                  }
                  onClick={() => void recordFinish()}
                >
                  finish dataset
                </button>
              </div>
            )}
          </>
        ) : (
          <button
            disabled={noInertia}
            title={
              noInertia
                ? "robot has no inertial data"
                : `stream physics live from this pose (${liveEngine})`
            }
            onClick={() => void startLive()}
          >
            ◉ Start live
          </button>
        )}
        {/* both outlive the session: a take can die with it, and a finished
            dataset is still worth opening after the session is gone */}
        {recHint && <p className="hint rec-hint">{recHint}</p>}
        {recDone && (
          <div className="rec-done">
            <p className="hint" title={recDone.root}>
              {recDone.episodes} episode{recDone.episodes === 1 ? "" : "s"} → {recDone.root}
            </p>
            <button
              title="browse the dataset in Data mode"
              onClick={() => void openRecordedDataset()}
            >
              open in Data
            </button>
          </div>
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
