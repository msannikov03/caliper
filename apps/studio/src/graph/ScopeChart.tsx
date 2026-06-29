import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

const WIDTH = 224;
const HEIGHT = 116;

/**
 * A tiny uPlot line chart for a Scope node. Reads ONLY the small t/y arrays the
 * executor already extracted (no per-frame work). Marked `nodrag nowheel` so the
 * React Flow canvas does not pan/zoom when interacting with the plot.
 */
export function ScopeChart({ t, y, label }: { t: number[]; y: number[]; label: string }) {
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
          stroke: "#6c6c7a",
          grid: { stroke: "#1b1b22", width: 1 },
          ticks: { stroke: "#1b1b22", width: 1 },
          font: "9px Inter, sans-serif",
          size: 22,
        },
        {
          stroke: "#6c6c7a",
          grid: { stroke: "#1b1b22", width: 1 },
          ticks: { stroke: "#1b1b22", width: 1 },
          font: "9px Inter, sans-serif",
          size: 34,
        },
      ],
      series: [
        {},
        { label: labelRef.current, stroke: "#36c6d4", width: 1.5, points: { show: false } },
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
    plot.current?.setData([t, y] as uPlot.AlignedData);
  }, [t, y]);

  return <div ref={host} className="scope-chart nodrag nowheel" />;
}
