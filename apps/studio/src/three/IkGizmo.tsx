import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { PivotControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { useStore } from "../store";
import { DISPLAY_UP, DISPLAY_UP_INV } from "../coords";

/**
 * Drag-to-IK handle anchored at the tip frame.
 *
 * It sits inside a group carrying DISPLAY_UP (the same Z-up→Y-up rotation as the
 * robot), so the controlled `matrix` we feed PivotControls is the tip pose in
 * URDF world (T_tip). PivotControls reports its drag world matrix `w` =
 * DISPLAY_UP · T_target; we recover the URDF-world target with DISPLAY_UP_INV
 * and hand it to the engine. OrbitControls is disabled while dragging.
 */
export function IkGizmo() {
  const robot = useStore((s) => s.robot);
  const frames = useStore((s) => s.frames);
  const playing = useStore((s) => s.playing);
  const mode = useStore((s) => s.mode);
  const solveIkGoverned = useStore((s) => s.solveIkGoverned);
  const controls = useThree((s) => s.controls) as unknown as
    | { enabled: boolean }
    | undefined;

  const groupMatrix = useMemo(() => DISPLAY_UP.clone(), []);
  const raf = useRef(0);
  const lastWorld = useRef(new THREE.Matrix4());
  const tmp = useMemo(() => new THREE.Matrix4(), []);

  // cancel any queued drag-follow rAF if we unmount mid-drag.
  useEffect(() => () => {
    if (raf.current) cancelAnimationFrame(raf.current);
  }, []);

  const tip = robot?.tip ?? -1;
  const tipMat = tip >= 0 ? frames[tip] : undefined;

  // controlled pivot transform = tip pose in URDF world (group applies DISPLAY_UP).
  const pivotMatrix = useMemo(() => {
    const m = new THREE.Matrix4();
    if (tipMat && tipMat.length === 16) m.fromArray(tipMat);
    return m;
  }, [tipMat]);

  // gizmo hidden during playback + in simulate mode (the sim owns the pose)
  if (!robot || !tipMat || playing || mode === "simulate") return null;

  const queue = (w: THREE.Matrix4) => {
    if (raf.current) return;
    const world = w.clone();
    raf.current = requestAnimationFrame(() => {
      raf.current = 0;
      // URDF-world target = DISPLAY_UP⁻¹ · (three-world gizmo matrix)
      tmp.copy(DISPLAY_UP_INV).multiply(world);
      void solveIkGoverned(tmp.toArray(), false); // damped live-follow, no snap
    });
  };

  return (
    <group matrixAutoUpdate={false} matrix={groupMatrix}>
      <PivotControls
        matrix={pivotMatrix}
        autoTransform={false}
        disableScaling
        depthTest={false}
        scale={0.18}
        lineWidth={2.5}
        axisColors={["#ff5a5a", "#5aff7a", "#5a9bff"]}
        onDragStart={() => {
          if (controls) controls.enabled = false;
        }}
        onDrag={(_l, _dl, w) => {
          lastWorld.current.copy(w);
          queue(w);
        }}
        onDragEnd={() => {
          if (raf.current) {
            cancelAnimationFrame(raf.current);
            raf.current = 0;
          }
          if (controls) controls.enabled = true;
          // exact final snap from the last drag world matrix (same path as queue)
          tmp.copy(DISPLAY_UP_INV).multiply(lastWorld.current);
          void solveIkGoverned(tmp.toArray(), true);
        }}
      />
    </group>
  );
}
