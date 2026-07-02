import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { useStore } from "../store";
import "./panels.css";

/** Filename (last path segment) for display; full path shown in the title tooltip. */
function baseName(p: string): string {
  return p.split(/[\\/]/).pop() || p;
}

export function Toolbar({ version }: { version: string }) {
  const loadRobot = useStore((s) => s.loadRobot);
  const robot = useStore((s) => s.robot);
  const loading = useStore((s) => s.loading);
  const recentUrdfs = useStore((s) => s.recentUrdfs);
  const addRecent = useStore((s) => s.addRecent);
  const removeRecent = useStore((s) => s.removeRecent);
  const loadRecent = useStore((s) => s.loadRecent);
  const [fixtures, setFixtures] = useState<[string, string][]>([]);
  const [sel, setSel] = useState<string>("");

  // hydrate persisted recents once on mount (prunes missing files async)
  useEffect(() => {
    loadRecent();
  }, [loadRecent]);

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

  // Load `path`, recording/pruning it in the recents list. `record` re-adds a
  // successfully-loaded path to the top (used by Open… and reselecting a recent);
  // a failed load of a recent path prunes it (the file likely moved/vanished).
  async function selectRobot(path: string, record: boolean) {
    setSel(path);
    const wasRecent = useStore.getState().recentUrdfs.includes(path);
    await loadRobot(path); // surfaces any CompileError in the shared error UI
    const st = useStore.getState();
    if (st.robot && !st.error) {
      if (record) addRecent(path);
    } else if (wasRecent) {
      removeRecent(path);
    }
  }

  async function openUrdf() {
    const picked = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "URDF", extensions: ["urdf", "xacro"] }],
    });
    if (typeof picked !== "string") return; // dialog cancelled
    await selectRobot(picked, true);
  }

  return (
    <header className="topbar" data-tauri-drag-region>
      <span className="brand">Caliper Studio</span>
      <button className="btn ghost" disabled={loading} onClick={() => void openUrdf()}>
        Open URDF…
      </button>
      <select
        className="fixture-select"
        value={sel}
        disabled={loading || (fixtures.length === 0 && recentUrdfs.length === 0)}
        onChange={(e) => {
          // reselecting a recent re-adds (bumps) it; samples are not recorded.
          const isRecent = recentUrdfs.includes(e.target.value);
          void selectRobot(e.target.value, isRecent);
        }}
      >
        <optgroup label="Samples">
          {fixtures.map(([name, path]) => (
            <option key={path} value={path} title={path}>
              {name}
            </option>
          ))}
        </optgroup>
        {recentUrdfs.length > 0 && (
          <optgroup label="Recent">
            {recentUrdfs.map((path) => (
              <option key={path} value={path} title={path}>
                {baseName(path)}
              </option>
            ))}
          </optgroup>
        )}
      </select>
      <button
        className="btn ghost"
        disabled={loading || !sel}
        onClick={() => sel && void selectRobot(sel, recentUrdfs.includes(sel))}
      >
        Reload
      </button>
      <span className="engine">
        {robot ? `· ${robot.name} · ` : ""}engine v{version}
      </span>
    </header>
  );
}
