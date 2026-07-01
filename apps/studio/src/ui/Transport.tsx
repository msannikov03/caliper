import { useStore } from "../store";
import type { SimTrajectoryDto } from "../store";
import "./panels.css";

/** seconds -> M:SS.cc, the instrument transport clock */
function clock(t: number): string {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}

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
      <button
        className="play"
        aria-label={playing ? "pause" : "play"}
        onClick={() => (playing ? pause() : play())}
      >
        {playing ? (
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
            <rect x="2.5" y="2" width="2.6" height="8" rx="0.6" fill="#fff" />
            <rect x="6.9" y="2" width="2.6" height="8" rx="0.6" fill="#fff" />
          </svg>
        ) : (
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
            <path d="M3 2 L10 6 L3 10 Z" fill="#fff" />
          </svg>
        )}
      </button>
      <input
        type="range"
        min={0}
        max={clip.duration}
        step={0.001}
        value={playhead}
        onChange={(e) => seek(parseFloat(e.target.value))}
      />
      <span className="tp-time">
        {clock(playhead)} <span className="tot">/ {clock(clip.duration)}</span>
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
      <span className="tp-rate">1.0×</span>
      <button aria-label="clear clip" title="clear clip" onClick={clearTraj}>
        <svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true">
          <path
            d="M3 3 L9 9 M9 3 L3 9"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinecap="round"
          />
        </svg>
      </button>
    </aside>
  );
}
