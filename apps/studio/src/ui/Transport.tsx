import { useStore } from "../store";

export function Transport() {
  const traj = useStore((s) => s.traj);
  const playing = useStore((s) => s.playing);
  const playhead = useStore((s) => s.playhead);
  const play = useStore((s) => s.play);
  const pause = useStore((s) => s.pause);
  const seek = useStore((s) => s.seek);
  const clearTraj = useStore((s) => s.clearTraj);
  if (!traj) return null;
  const within = traj.maxJerkRatio <= 1.0;
  return (
    <aside className="transport">
      <button onClick={() => (playing ? pause() : play())}>{playing ? "⏸" : "▶"}</button>
      <input
        type="range"
        min={0}
        max={traj.duration}
        step={0.001}
        value={playhead}
        onChange={(e) => seek(parseFloat(e.target.value))}
      />
      <span className="time">
        {playhead.toFixed(2)}/{traj.duration.toFixed(2)}s
      </span>
      {!traj.ok && (
        <span className="badge bad" title="best-effort prefix">
          stopped @ {(traj.reached * 100).toFixed(0)}%
        </span>
      )}
      <span className={within ? "badge ok" : "badge bad"} title="max sampled jerk / limit">
        {within ? "within limits" : `jerk ×${traj.maxJerkRatio.toFixed(2)}`}
      </span>
      <button onClick={clearTraj}>✕</button>
    </aside>
  );
}
