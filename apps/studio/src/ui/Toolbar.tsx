import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStore } from "../store";

export function Toolbar({ version }: { version: string }) {
  const loadRobot = useStore((s) => s.loadRobot);
  const robot = useStore((s) => s.robot);
  const loading = useStore((s) => s.loading);
  const [fixtures, setFixtures] = useState<[string, string][]>([]);
  const [sel, setSel] = useState<string>("");

  useEffect(() => {
    invoke<[string, string][]>("fixtures")
      .then((f) => {
        setFixtures(f);
        if (f.length) {
          setSel(f[0][1]);
          void loadRobot(f[0][1]); // auto-load the showcase on startup
        }
      })
      .catch(() => setFixtures([]));
  }, [loadRobot]);

  return (
    <header className="topbar" data-tauri-drag-region>
      <span className="brand">Caliper Studio</span>
      <select
        className="fixture-select"
        value={sel}
        disabled={loading || fixtures.length === 0}
        onChange={(e) => {
          setSel(e.target.value);
          void loadRobot(e.target.value);
        }}
      >
        {fixtures.map(([name, path]) => (
          <option key={path} value={path}>
            {name}
          </option>
        ))}
      </select>
      <span className="engine">
        {robot ? `· ${robot.name} · ` : ""}engine v{version}
      </span>
    </header>
  );
}
