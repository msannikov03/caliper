import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

const WIDTH = 224;
const HEIGHT = 116;

/**
 * A tiny uPlot line chart for a Scope node. Reads ONLY the small t/y arrays the
 * executor already extracted (no per-frame work). Marked `nodrag nowheel` so the
 * React Flow canvas does not pan/zoom when interacting with the plot.
 *
 * When `live` is set the chart "streams" the already-computed series in: the
 * x-axis stays at full extent while the y-trace is revealed left-to-right via
 * `setData` on a requestAnimationFrame timer (samples beyond the cursor are
 * `null`, so uPlot just leaves a gap). This is purely a playback affordance over
 * data the backend already produced — no extra compute, no new IPC.
 *
 * NOTE: the live path below is BUILD-CHECKED ONLY. The Studio app is never run in
 * this workflow, so the streaming animation has not been runtime-verified; it is
 * written to be type-safe and to degrade to the static plot if anything is off.
 */
export function ScopeChart({
  t,
  y,
  label,
  live = false,
}: {
  t: number[];
  y: number[];
  label: string;
  live?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const labelRef = useRef(label);
  labelRef.current = label;

  useEffect(() => {
    if (!host.current) return;
    const opts: uPlot.Options = {
      width: WIDTH,
      height: HEIGHT,
      legend: { show: false },
      cursor: { show: false },
      scales: { x: { time: false } },
      axes: [
        {
          stroke: "#8a9099",
          grid: { stroke: "rgba(255,255,255,0.05)", width: 1 },
          ticks: { stroke: "rgba(255,255,255,0.08)", width: 1 },
          font: "9px 'JetBrains Mono', ui-monospace, monospace",
          size: 22,
        },
        {
          stroke: "#8a9099",
          grid: { stroke: "rgba(255,255,255,0.05)", width: 1 },
          ticks: { stroke: "rgba(255,255,255,0.08)", width: 1 },
          font: "9px 'JetBrains Mono', ui-monospace, monospace",
          size: 34,
        },
      ],
      series: [
        {},
        {
          label: labelRef.current,
          stroke: "#7c82ff",
          fill: "rgba(124,130,255,0.10)",
          width: 1.5,
          points: { show: false },
        },
      ],
    };
    plot.current = new uPlot(opts, [t, y] as uPlot.AlignedData, host.current);
    return () => {
      plot.current?.destroy();
      plot.current = null;
    };
    // build once; data updates ride the second effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const p = plot.current;
    if (!p) return;
    // Static render (the unchanged batch path): paint the full series at once.
    if (!live || t.length === 0) {
      p.setData([t, y] as uPlot.AlignedData);
      return;
    }
    // --- live streaming reveal (build-checked only; not runtime-verified) ---
    // Reveal `y` left-to-right over a short wall-clock window scaled by sample
    // count (clamped), holding the x-axis fixed by nulling un-revealed samples.
    const n = t.length;
    const totalMs = Math.min(2000, Math.max(400, n * 6));
    const startedAt = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const frac = Math.min(1, (now - startedAt) / totalMs);
      const cursor = Math.max(1, Math.ceil(frac * n));
      const yLive: (number | null)[] = y.map((v, i) => (i < cursor ? v : null));
      p.setData([t, yLive] as uPlot.AlignedData);
      if (frac < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [t, y, live]);

  return <div ref={host} className="scope-chart nodrag nowheel" />;
}
