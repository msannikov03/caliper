import { useLayoutEffect, useRef } from "react";
import * as THREE from "three";
import { useStore } from "../store";

/** Traced Cartesian tip path of the loaded trajectory. Rendered INSIDE the
 *  RobotView DISPLAY_UP group, so URDF-world tipPath points land in display space. */
export function TipPath() {
  const traj = useStore((s) => s.traj);
  const ref = useRef<THREE.Line>(null!);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const pts: number[] = [];
    if (traj) for (const p of traj.tipPath) pts.push(p[0], p[1], p[2]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    ref.current.geometry.dispose();
    ref.current.geometry = geo;
    return () => geo.dispose(); // free the GPU buffer when the path is replaced/unmounted
  }, [traj]);
  if (!traj) return null;
  return (
    <line ref={ref as never}>
      <bufferGeometry />
      <lineBasicMaterial color="#36c6d4" transparent opacity={0.9} depthTest={false} />
    </line>
  );
}
