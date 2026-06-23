import { useStore } from "../store";
import type { SimTrajectoryDto } from "../store";

export function Transport() {
  const clip = useStore((s) => s.simTraj ?? s.traj); // active clip
  const playing = useStore((s) => s.playing);
  const playhead = useStore((s) => s.playhead);
  const play = useStore((s) => s.play);
  const pause = useStore((s) => s.pause);
  const seek = useStore((s) => s.seek);
  const clearTraj = useStore((s) => s.clearTraj);
  if (!clip) return null;
  const isSim = clip.kind === "sim";
  const within = clip.maxJerkRatio <= 1.0;
  const drift = isSim ? (clip as SimTrajectoryDto).energyDrift : 0;
  return (
    <aside className="transport">
      <button onClick={() => (playing ? pause() : play())}>{playing ? "⏸" : "▶"}</button>
      <input
        type="range"
        min={0}
        max={clip.duration}
        step={0.001}
        value={playhead}
        onChange={(e) => seek(parseFloat(e.target.value))}
      />
      <span className="time">
        {playhead.toFixed(2)}/{clip.duration.toFixed(2)}s
      </span>
      {isSim ? (
        <span className={drift < 1e-3 ? "badge ok" : "badge"} title="energy drift over the sim">
          energy drift {(drift * 100).toFixed(3)}%
        </span>
      ) : (
        <>
          {!clip.ok && (
            <span className="badge bad" title="best-effort prefix">
              stopped @ {(clip.reached * 100).toFixed(0)}%
            </span>
          )}
          <span className={within ? "badge ok" : "badge bad"} title="max sampled jerk / limit">
            {within ? "within limits" : `jerk ×${clip.maxJerkRatio.toFixed(2)}`}
          </span>
        </>
      )}
      <button onClick={clearTraj}>✕</button>
    </aside>
  );
}
