// ============================================================
// Palette.tsx — the ⌘K command palette. A floating instrument
// over a dimmed backdrop; a pure overlay, so the persistent GL
// <Canvas> underneath is never unmounted. All command content,
// gating and ranking live headlessly in ../commands.ts — this
// file only renders and drives keyboard/mouse selection.
// ============================================================

import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store";
import { buildCommands, filterCommands } from "../commands";
import type { Command } from "../commands";
import { openUrdf, selectRobot } from "./Toolbar";
import "./palette.css";

export function Palette({ onClose }: { onClose: () => void }) {
  const [query, setQuery] = useState("");
  const [sel, setSel] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // the store slices the command list depends on (rebuild on change)
  const robot = useStore((s) => s.robot);
  const mode = useStore((s) => s.mode);
  const fixtures = useStore((s) => s.fixtures);
  const recents = useStore((s) => s.recentUrdfs);
  const poses = useStore((s) => s.poses);
  const urdfPath = useStore((s) => s.urdfPath);

  const commands = useMemo(() => {
    // every action wraps an existing store/toolbar operation — no new backend calls
    const home = () => new Array(useStore.getState().robot?.ndof ?? 0).fill(0);
    const all = buildCommands({
      fixtures,
      recents,
      poses: poses.map((p) => p.name),
      mode,
      robotLoaded: robot !== null,
      hasInertia: robot?.hasInertia ?? false,
      urdfPath,
      actions: {
        openUrdf: () => void openUrdf(),
        openPath: (path, record) => void selectRobot(path, record),
        setMode: (m) => useStore.getState().setMode(m),
        planHome: () => void useStore.getState().planMoveJ(home()),
        planToPose: (name) => void useStore.getState().planMoveToPose(name),
        driveHome: () => void useStore.getState().runControl(home()),
        gravityDrop: () => void useStore.getState().runGravityDrop(),
        planRrtHome: () => void useStore.getState().runPlan(home()),
        checkCollision: () => void useStore.getState().checkCollision(null),
        runGraph: () => void useStore.getState().runGraph(),
        validateGraph: () => void useStore.getState().validateGraph(),
      },
    });
    return filterCommands(query, all);
  }, [query, fixtures, recents, poses, mode, robot, urdfPath]);

  useEffect(() => inputRef.current?.focus(), []);

  // whenever the filtered list changes, land the highlight on the first runnable row
  useEffect(() => {
    setSel(Math.max(0, commands.findIndex((c) => c.enabled)));
  }, [commands]);

  // keep the highlighted row in view during keyboard nav
  useEffect(() => {
    listRef.current?.querySelector(".sel")?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  /// Move the highlight to the next/previous ENABLED row, wrapping.
  function step(dir: 1 | -1) {
    if (commands.length === 0) return;
    let i = sel;
    for (let n = 0; n < commands.length; n++) {
      i = (i + dir + commands.length) % commands.length;
      if (commands[i].enabled) break;
    }
    setSel(i);
  }

  function runCmd(c: Command) {
    if (!c.enabled) return;
    onClose(); // close first — a command may move focus or switch modes
    c.run();
  }

  // The palette owns Arrow/Enter/Esc (stopPropagation so the global keymap
  // never sees them); everything else — ⌘K, ⌘O, ⌘1… — falls through to it.
  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      e.stopPropagation();
      step(e.key === "ArrowDown" ? 1 : -1);
    } else if (e.key === "Enter") {
      e.preventDefault();
      e.stopPropagation();
      const c = commands[sel];
      if (c) runCmd(c);
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      onClose();
    }
  }

  return (
    <div
      className="cmdk-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="cmdk" onKeyDown={onKeyDown}>
        <div className="cmdk-input">
          <span className="cmdk-glyph">⌘</span>
          <input
            ref={inputRef}
            autoFocus
            spellCheck={false}
            placeholder="Type a command…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <span className="cmdk-esc">esc</span>
        </div>
        <div className="cmdk-list" ref={listRef}>
          {commands.length === 0 && <div className="cmdk-empty">no matching command</div>}
          {commands.map((c, i) => (
            <Fragment key={c.id}>
              {(i === 0 || commands[i - 1].section !== c.section) && (
                <div className="cmdk-section">{c.section}</div>
              )}
              <div
                className={`cmdk-row${i === sel ? " sel" : ""}${c.enabled ? "" : " off"}`}
                onMouseEnter={() => c.enabled && setSel(i)}
                onMouseDown={(e) => e.preventDefault()} // keep the input focused
                onClick={() => runCmd(c)}
              >
                <span className="cmdk-title">{c.title}</span>
                {c.hint && <span className="cmdk-hint">{c.hint}</span>}
              </div>
            </Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}
