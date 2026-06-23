import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useStore } from "../store";
import type { FrameInfo } from "../store";
import { DISPLAY_UP } from "../coords";
import { Ellipsoid } from "./Ellipsoid";
import { TipPath } from "./TipPath";

const AX_X = "#ff5a5a";
const AX_Y = "#5aff7a";
const AX_Z = "#5a9bff";
const ACCENT = "#36c6d4";
const ROD = "#3a3d44";

/** Three short axis lines (frame-local RGB triad). */
function Triad({ size = 0.04, tip = false }: { size?: number; tip?: boolean }) {
  const obj = useMemo(() => {
    const grp = new THREE.Group();
    const mk = (dir: [number, number, number], c: string) => {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(...dir).multiplyScalar(size),
      ]);
      const mat = new THREE.LineBasicMaterial({
        color: c,
        transparent: !tip,
        opacity: tip ? 1 : 0.45,
      });
      return new THREE.Line(geo, mat);
    };
    grp.add(mk([1, 0, 0], AX_X), mk([0, 1, 0], AX_Y), mk([0, 0, 1], AX_Z));
    return grp;
  }, [size, tip]);
  return <primitive object={obj} />;
}

/** Joint marker oriented to the frame-local joint axis (parent group already
 *  applies the frame's world rotation). Revolute → hinge knuckle; prismatic → rail. */
function JointMarker({ frame }: { frame: FrameInfo }) {
  const quat = useMemo(() => {
    if (!frame.axis) return new THREE.Quaternion();
    const a = new THREE.Vector3(...frame.axis).normalize();
    // cylinder/box local long axis is +Y; rotate it onto the joint axis.
    return new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), a);
  }, [frame.axis]);

  if (frame.jointIndex < 0 || !frame.axis) return null;
  return (
    <group quaternion={quat}>
      {frame.jointKind === "prismatic" ? (
        <mesh>
          <boxGeometry args={[0.012, 0.05, 0.012]} />
          <meshStandardMaterial
            color={ACCENT}
            metalness={0.6}
            roughness={0.3}
            emissive="#0c3b42"
            emissiveIntensity={0.4}
          />
        </mesh>
      ) : (
        <mesh>
          <cylinderGeometry args={[0.016, 0.016, 0.03, 20]} />
          <meshStandardMaterial
            color={ACCENT}
            metalness={0.6}
            roughness={0.3}
            emissive="#0c3b42"
            emissiveIntensity={0.4}
          />
        </mesh>
      )}
    </group>
  );
}

/** One frame: a group whose LOCAL matrix is the engine's URDF-world pose. */
function FrameNode({ index }: { index: number }) {
  const ref = useRef<THREE.Group>(null!);
  const frame = useStore((s) => s.robot!.frames[index]);
  const m = useStore((s) => s.frames[index]);
  const isTip = useStore((s) => s.robot!.tip === index);

  useLayoutEffect(() => {
    if (!ref.current || !m || m.length !== 16) return;
    ref.current.matrix.fromArray(m); // THREE.Matrix4 is column-major: direct.
    ref.current.matrixWorldNeedsUpdate = true;
  }, [m]);

  return (
    <group ref={ref} matrixAutoUpdate={false}>
      <Triad tip={isTip} size={isTip ? 0.06 : 0.04} />
      <JointMarker frame={frame} />
    </group>
  );
}

/** Rods: one line segment per non-root frame, from its origin to its parent's. */
function Rods() {
  const frames = useStore((s) => s.frames);
  const info = useStore((s) => s.robot!.frames);
  const ref = useRef<THREE.LineSegments>(null!);

  useLayoutEffect(() => {
    if (!ref.current || frames.length !== info.length) return;
    const pts: number[] = [];
    for (let i = 0; i < info.length; i++) {
      const p = info[i].parent;
      if (p < 0 || !frames[i] || !frames[p]) continue;
      const a = frames[i];
      const b = frames[p];
      // translation = column 3 of a column-major 4x4 = indices 12,13,14.
      pts.push(a[12], a[13], a[14], b[12], b[13], b[14]);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    ref.current.geometry.dispose();
    ref.current.geometry = geo;
  }, [frames, info]);

  return (
    <lineSegments ref={ref}>
      <bufferGeometry />
      <lineBasicMaterial color={ROD} linewidth={2} />
    </lineSegments>
  );
}

/** Machined base plate at the root, on the floor. */
function BasePlate() {
  return (
    <mesh position={[0, 0, 0.007]} rotation={[Math.PI / 2, 0, 0]}>
      <cylinderGeometry args={[0.09, 0.09, 0.014, 48]} />
      <meshStandardMaterial color="#17171c" metalness={0.9} roughness={0.35} />
    </mesh>
  );
}

/** The whole robot, rendered inside the Z-up→Y-up display group. */
export function RobotView() {
  const robot = useStore((s) => s.robot);
  const matrix = useMemo(() => DISPLAY_UP.clone(), []);
  if (!robot) return null;
  return (
    <group matrixAutoUpdate={false} matrix={matrix}>
      <BasePlate />
      <Rods />
      <TipPath />
      <Ellipsoid />
      {robot.frames.map((_, i) => (
        <FrameNode key={i} index={i} />
      ))}
    </group>
  );
}
