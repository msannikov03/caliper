import { useEffect } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { useStore } from "../store";
import { baseName } from "../commands";
import "./panels.css";

// Load `path`, recording/pruning it in the recents list. `record` re-adds a
// successfully-loaded path to the top (used by Open… and reselecting a recent);
// a failed load of a recent path prunes it (the file likely moved/vanished).
// Module-scope on purpose: the toolbar, the ⌘K palette and the ⌘O shortcut all
// share this ONE implementation.
export async function selectRobot(path: string, record: boolean): Promise<void> {
  const wasRecent = useStore.getState().recentUrdfs.includes(path);
  await useStore.getState().loadRobot(path); // surfaces any CompileError in the shared error UI
  const st = useStore.getState();
  if (st.robot && !st.error) {
    if (record) st.addRecent(path);
  } else if (wasRecent) {
    st.removeRecent(path);
  }
}

/** Native URDF picker → load + record. The Open… button, ⌘O and the palette. */
export async function openUrdf(): Promise<void> {
  const picked = await open({
    multiple: false,
    directory: false,
    filters: [{ name: "URDF", extensions: ["urdf", "xml"] }],
  });
  if (typeof picked !== "string") return; // dialog cancelled
  await selectRobot(picked, true);
}

export function Toolbar({ version, onPalette }: { version: string; onPalette: () => void }) {
  const robot = useStore((s) => s.robot);
  const loading = useStore((s) => s.loading);
  const recentUrdfs = useStore((s) => s.recentUrdfs);
  const loadRecent = useStore((s) => s.loadRecent);
  const fixtures = useStore((s) => s.fixtures);
  const loadFixtures = useStore((s) => s.loadFixtures);
  // the select mirrors whatever the store last (attempted to) load
  const sel = useStore((s) => s.urdfPath) ?? "";

  // hydrate persisted recents once on mount (prunes missing files async)
  useEffect(() => {
    loadRecent();
  }, [loadRecent]);

  useEffect(() => {
    void loadFixtures().then(() => {
      if (!useStore.getState().urdfPath) {
        // resume the previous session (robot + pose + mode) when one restores
        // cleanly; otherwise this falls back to auto-loading the first sample.
        void useStore.getState().restoreSession(selectRobot);
      }
    });
  }, [loadFixtures]);

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
      <button className="kbd-chip" title="Command palette (⌘K)" onClick={onPalette}>
        ⌘K
      </button>
    </header>
  );
}
