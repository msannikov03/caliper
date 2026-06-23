import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useStore } from "../store";
import { severity, rampColor } from "../sevColor";

const GAIN = 0.12; // cosmetic: σ (m/rad·m) -> scene metres. Single-robot view only.

export function Ellipsoid() {
  const report = useStore((s) => s.report);
  const ref = useRef<THREE.Mesh>(null!);
  const mat = useMemo(() => new THREE.Matrix4(), []);

  useLayoutEffect(() => {
    if (!ref.current || !report) return;
    const [ax, ay, az] = report.ellipsoidAxes; // columns = unit principal axes
    const r = report.ellipsoidRadii;
    // THREE.Matrix4.set is ROW-major; basis COLUMNS go down each literal column.
    mat.set(
      ax[0] * r[0] * GAIN, ay[0] * r[1] * GAIN, az[0] * r[2] * GAIN, report.tipWorld[0],
      ax[1] * r[0] * GAIN, ay[1] * r[1] * GAIN, az[1] * r[2] * GAIN, report.tipWorld[1],
      ax[2] * r[0] * GAIN, ay[2] * r[1] * GAIN, az[2] * r[2] * GAIN, report.tipWorld[2],
      0, 0, 0, 1,
    );
    ref.current.matrix.copy(mat);
    ref.current.matrixWorldNeedsUpdate = true;
  }, [report, mat]);

  if (!report) return null;
  const f = severity(report.sigmaMin, report.epsActivate);
  const c = rampColor(f);
  return (
    <mesh ref={ref} matrixAutoUpdate={false} renderOrder={2}>
      <sphereGeometry args={[1, 48, 32]} />
      <meshStandardMaterial
        color={c}
        transparent
        opacity={0.28}
        depthWrite={false}
        emissive={c}
        emissiveIntensity={0.15}
        roughness={0.5}
        metalness={0}
      />
    </mesh>
  );
}
