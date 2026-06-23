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
import "./App.css";

function SceneChrome() {
  return (
    <>
      <color attach="background" args={["#0a0a0c"]} />
      {/* hand-rolled studio lighting — no remote HDRI, so a strict CSP holds */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 8, 5]} intensity={1.1} />
      <directionalLight position={[-6, 3, -4]} intensity={0.4} color="#7fb0ff" />
      <directionalLight position={[3, -2, -6]} intensity={0.3} color="#36c6d4" />
      <Grid
        args={[24, 24]}
        cellSize={0.25}
        cellThickness={0.6}
        cellColor="#1b1b22"
        sectionSize={1}
        sectionThickness={1}
        sectionColor="#2c2c3a"
        fadeDistance={20}
        fadeStrength={1.2}
        infiniteGrid
      />
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
      <GizmoHelper alignment="bottom-right" margin={[72, 72]}>
        <GizmoViewport axisColors={["#ff5a5a", "#5aff7a", "#5a9bff"]} labelColor="#9a9aa6" />
      </GizmoHelper>
    </>
  );
}

export default function App() {
  const [version, setVersion] = useState("…");
  useEffect(() => {
    invoke<string>("engine_version")
      .then(setVersion)
      .catch(() => setVersion("offline"));
  }, []);

  return (
    <div className="app">
      <Toolbar version={version} />
      <main className="viewport">
        <Canvas camera={{ position: [0.7, 0.7, 0.7], fov: 50 }}>
          <SceneChrome />
          <RobotView />
          <IkGizmo />
        </Canvas>
        <JointPanel />
        <Hud />
        <SingularityHud />
      </main>
    </div>
  );
}
