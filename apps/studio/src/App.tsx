import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

function Scene() {
  return (
    <>
      <color attach="background" args={["#0a0a0c"]} />
      <ambientLight intensity={0.45} />
      <directionalLight position={[5, 8, 5]} intensity={1.2} castShadow />
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
      <header className="topbar" data-tauri-drag-region>
        <span className="brand">Caliper Studio</span>
        <span className="engine">engine v{version}</span>
      </header>
      <main className="viewport">
        <Canvas shadows camera={{ position: [1.6, 1.2, 1.6], fov: 50 }}>
          <Scene />
        </Canvas>
      </main>
    </div>
  );
}
