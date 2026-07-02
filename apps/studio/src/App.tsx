import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import { invoke } from "@tauri-apps/api/core";
import { RobotView } from "./three/RobotView";
import { IkGizmo } from "./three/IkGizmo";
import { Toolbar, openUrdf } from "./ui/Toolbar";
import { Palette } from "./ui/Palette";
import { MODE_TABS } from "./commands";
import { JointPanel } from "./ui/JointPanel";
import { Hud } from "./ui/Hud";
import { SingularityHud } from "./ui/SingularityHud";
import { Transport } from "./ui/Transport";
import { PosePanel } from "./ui/PosePanel";
import { SimulatePanel } from "./ui/SimulatePanel";
import { GraphEditor } from "./graph/GraphEditor";
import { useStore } from "./store";
import "./App.css";

function ModeTabs() {
  const mode = useStore((s) => s.mode);
  const setMode = useStore((s) => s.setMode);
  const robot = useStore((s) => s.robot);
  if (!robot) return null;
  // MODE_TABS is shared with the ⌘K palette so ⌘1…⌘4 always match tab order
  return (
    <div className="mode-tabs">
      {MODE_TABS.map((t) => {
        const disabled = t.id === "simulate" && !robot.hasInertia;
        return (
          <button
            key={t.id}
            disabled={disabled}
            title={disabled ? "no inertial data" : ""}
            className={mode === t.id ? "active" : ""}
            onClick={() => setMode(t.id)}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

function SceneChrome() {
  return (
    <>
      <color attach="background" args={["#0B0D10"]} />
      {/* hand-rolled studio lighting — no remote HDRI, so a strict CSP holds */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 8, 5]} intensity={1.1} />
      <directionalLight position={[-6, 3, -4]} intensity={0.4} color="#7c82ff" />
      <directionalLight position={[3, -2, -6]} intensity={0.3} color="#6e9bff" />
      <Grid
        args={[24, 24]}
        cellSize={0.25}
        cellThickness={0.6}
        cellColor="#1b1f26"
        sectionSize={1}
        sectionThickness={1}
        sectionColor="#2a2f3a"
        fadeDistance={20}
        fadeStrength={1.2}
        infiniteGrid
      />
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
      <GizmoHelper alignment="bottom-right" margin={[72, 72]}>
        <GizmoViewport axisColors={["#FF6470", "#56D98C", "#6E9BFF"]} labelColor="#8A9099" />
      </GizmoHelper>
    </>
  );
}

export default function App() {
  const [version, setVersion] = useState("…");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const mode = useStore((s) => s.mode);
  useEffect(() => {
    invoke<string>("engine_version")
      .then(setVersion)
      .catch(() => setVersion("offline"));
  }, []);

  // Global keymap (ONE window listener): ⌘K palette · ⌘O open URDF · ⌘1…⌘4
  // modes (ModeTabs order) · Esc closes the palette. macOS-first (metaKey),
  // ctrlKey accepted. Shortcuts are ignored while typing in a field — except
  // inside the palette, which stopPropagation()s the keys it consumes itself
  // and deliberately lets these chords fall through.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target instanceof HTMLElement ? e.target : null;
      const typing = !!t?.closest("input, textarea, [contenteditable]");
      if (typing && !t?.closest(".cmdk")) return;
      const mod = e.metaKey || e.ctrlKey;
      if (mod && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      } else if (mod && (e.key === "o" || e.key === "O")) {
        e.preventDefault();
        setPaletteOpen(false);
        void openUrdf();
      } else if (mod && e.key >= "1" && e.key <= String(MODE_TABS.length)) {
        const tab = MODE_TABS[Number(e.key) - 1];
        const st = useStore.getState();
        // mirror the ModeTabs gating: no robot → no tabs; simulate needs inertia
        if (st.robot && !(tab.id === "simulate" && !st.robot.hasInertia)) {
          e.preventDefault();
          st.setMode(tab.id);
        }
      } else if (e.key === "Escape") {
        setPaletteOpen(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const isGraph = mode === "graph";

  return (
    <div className="app">
      <Toolbar version={version} onPalette={() => setPaletteOpen(true)} />
      {/* segmented mode switch floats centered over the command bar */}
      <ModeTabs key="modetabs" />
      {/*
        Single persistent <Canvas>: in Graph mode it docks to the right as the live
        preview the View sink drives; otherwise it fills the viewport. The GL canvas
        is NEVER unmounted on a mode switch — only resized via CSS.
      */}
      <main className={`viewport${isGraph ? " graph-mode" : ""}`}>
        {isGraph && (
          <div className="graph-pane" key="graphpane">
            <GraphEditor />
          </div>
        )}
        {/* keyed so inserting .graph-pane never reconciles/remounts the GL Canvas */}
        <div className="gl-stage" key="glstage">
          <Canvas camera={{ position: [0.7, 0.7, 0.7], fov: 50 }}>
            <SceneChrome />
            <RobotView />
            {!isGraph && <IkGizmo />}
          </Canvas>
          {!isGraph && <JointPanel />}
          {!isGraph && <PosePanel />}
          {!isGraph && <SimulatePanel />}
          <Hud />
          <SingularityHud />
          <Transport />
        </div>
      </main>
      {/* overlay only — mounting/unmounting it never touches the GL Canvas */}
      {paletteOpen && <Palette onClose={() => setPaletteOpen(false)} />}
    </div>
  );
}
