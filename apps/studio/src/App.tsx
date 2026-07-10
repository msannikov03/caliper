import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import { invoke } from "@tauri-apps/api/core";
import { RobotView } from "./three/RobotView";
import { PropsLayer } from "./three/PropsLayer";
import { IkGizmo } from "./three/IkGizmo";
import { Toolbar, openUrdf } from "./ui/Toolbar";
import { Palette } from "./ui/Palette";
import { MODE_TABS, modeNeedsRobot } from "./commands";
import { JointPanel } from "./ui/JointPanel";
import { Hud } from "./ui/Hud";
import { SingularityHud } from "./ui/SingularityHud";
import { Transport } from "./ui/Transport";
import { PosePanel } from "./ui/PosePanel";
import { SimulatePanel } from "./ui/SimulatePanel";
import { GraphEditor } from "./graph/GraphEditor";
import { DataMode } from "./data/DataMode";
import { useStore } from "./store";
import "./App.css";

function ModeTabs() {
  const mode = useStore((s) => s.mode);
  const setMode = useStore((s) => s.setMode);
  const robot = useStore((s) => s.robot);
  // MODE_TABS is shared with the ⌘K palette so ⌘1…⌘5 always match tab order.
  // The tabs render even with NO robot loaded (least-invasive change from the
  // old early-return): robot-bound tabs just disable, so Data — which browses
  // datasets robot-free — stays reachable.
  return (
    <div className="mode-tabs">
      {MODE_TABS.map((t) => {
        const noRobot = modeNeedsRobot(t.id) && !robot;
        const noInertia = t.id === "simulate" && !!robot && !robot.hasInertia;
        const disabled = noRobot || noInertia;
        return (
          <button
            key={t.id}
            disabled={disabled}
            title={noRobot ? "no robot loaded" : noInertia ? "no inertial data" : ""}
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

  // Global keymap (ONE window listener): ⌘K palette · ⌘O open URDF · ⌘1…⌘5
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
        // single-digit string compare holds while MODE_TABS.length <= 9 (now 5)
        const tab = MODE_TABS[Number(e.key) - 1];
        const st = useStore.getState();
        // mirror the ModeTabs gating: robot-bound tabs need a robot (Data does
        // not); simulate additionally needs inertia
        const okRobot = !modeNeedsRobot(tab.id) || !!st.robot;
        if (okRobot && !(tab.id === "simulate" && !st.robot?.hasInertia)) {
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
  const isData = mode === "data";
  const docked = isGraph || isData; // canvas docks right; jog/motion overlays hide

  return (
    <div className="app">
      <Toolbar version={version} onPalette={() => setPaletteOpen(true)} />
      {/* segmented mode switch floats centered over the command bar */}
      <ModeTabs key="modetabs" />
      {/*
        Single persistent <Canvas>: in Graph/Data mode it docks to the right (the
        live preview / a passive robot view); otherwise it fills the viewport. The
        GL canvas is NEVER unmounted on a mode switch — only resized via CSS.
      */}
      <main className={`viewport${isGraph ? " graph-mode" : ""}${isData ? " data-mode" : ""}`}>
        {isGraph && (
          <div className="graph-pane" key="graphpane">
            <GraphEditor />
          </div>
        )}
        {isData && (
          <div className="data-pane" key="datapane">
            <DataMode />
          </div>
        )}
        {/* keyed so inserting a side pane never reconciles/remounts the GL Canvas */}
        <div className="gl-stage" key="glstage">
          <Canvas camera={{ position: [0.7, 0.7, 0.7], fov: 50 }}>
            <SceneChrome />
            <RobotView />
            {/* contact-sim free props follow the same playback clock */}
            <PropsLayer />
            {!docked && <IkGizmo />}
          </Canvas>
          {!docked && <JointPanel />}
          {!docked && <PosePanel />}
          {!docked && <SimulatePanel />}
          {/* Data mode drives no clips and owns its own error banner — the robot
              HUD/transport overlays would only mislead over the docked preview */}
          {!isData && <Hud />}
          {!isData && <SingularityHud />}
          {!isData && <Transport />}
        </div>
      </main>
      {/* overlay only — mounting/unmounting it never touches the GL Canvas */}
      {paletteOpen && <Palette onClose={() => setPaletteOpen(false)} />}
    </div>
  );
}
