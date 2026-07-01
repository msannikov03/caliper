import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import { invoke } from "@tauri-apps/api/core";
import { RobotView } from "./three/RobotView";
import { IkGizmo } from "./three/IkGizmo";
import { Toolbar } from "./ui/Toolbar";
import { JointPanel } from "./ui/JointPanel";
import { Hud } from "./ui/Hud";
import { SingularityHud } from "./ui/SingularityHud";
import { Transport } from "./ui/Transport";
import { PosePanel } from "./ui/PosePanel";
import { SimulatePanel } from "./ui/SimulatePanel";
import { GraphEditor } from "./graph/GraphEditor";
import { useStore } from "./store";
import type { StudioMode } from "./store";
import "./App.css";

function ModeTabs() {
  const mode = useStore((s) => s.mode);
  const setMode = useStore((s) => s.setMode);
  const robot = useStore((s) => s.robot);
  if (!robot) return null;
  const tabs: { id: StudioMode; label: string; disabled?: boolean }[] = [
    { id: "jog", label: "Jog" },
    { id: "motion", label: "Motion" },
    { id: "simulate", label: "Simulate", disabled: !robot.hasInertia },
    { id: "graph", label: "Graph" },
  ];
  return (
    <div className="mode-tabs">
      {tabs.map((t) => (
        <button
          key={t.id}
          disabled={t.disabled}
          title={t.disabled ? "no inertial data" : ""}
          className={mode === t.id ? "active" : ""}
          onClick={() => setMode(t.id)}
        >
          {t.label}
        </button>
      ))}
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
  const mode = useStore((s) => s.mode);
  useEffect(() => {
    invoke<string>("engine_version")
      .then(setVersion)
      .catch(() => setVersion("offline"));
  }, []);

  const isGraph = mode === "graph";

  return (
    <div className="app">
      <Toolbar version={version} />
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
    </div>
  );
}
