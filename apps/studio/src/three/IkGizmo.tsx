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
 *
 * Two destinations for that target:
 *   - normally solveIkGoverned, which poses the robot directly;
 *   - during a LIVE session driveTipLive, which turns it into the session's
 *     hold target and never touches `q` — the stream owns the pose, so the two
 *     can't fight. The handle also stops tracking the streamed tip for the
 *     duration of a live drag (the arm is still converging on it; a handle that
 *     chased the lagging tip would slide out from under the cursor).
 */
export function IkGizmo() {
  const robot = useStore((s) => s.robot);
  const frames = useStore((s) => s.frames);
  const playing = useStore((s) => s.playing);
  const mode = useStore((s) => s.mode);
  const live = useStore((s) => s.live);
  const solveIkGoverned = useStore((s) => s.solveIkGoverned);
  const driveTipLive = useStore((s) => s.driveTipLive);
  const controls = useThree((s) => s.controls) as unknown as
    | { enabled: boolean }
    | undefined;

  const groupMatrix = useMemo(() => DISPLAY_UP.clone(), []);
  const raf = useRef(0);
  const lastWorld = useRef(new THREE.Matrix4());
  const tmp = useMemo(() => new THREE.Matrix4(), []);
  // live drag only: the handle pose held still while the arm catches up
  const dragging = useRef(false);
  const held = useRef<THREE.Matrix4 | null>(null);
  const refs: DragRefs = { raf, dragging, held };

  const tip = robot?.tip ?? -1;
  const tipMat = tip >= 0 ? frames[tip] : undefined;

  // controlled pivot transform = tip pose in URDF world (group applies DISPLAY_UP).
  const pivotMatrix = useMemo(() => {
    const m = new THREE.Matrix4();
    if (tipMat && tipMat.length === 16) m.fromArray(tipMat);
    return m;
  }, [tipMat]);

  // hidden during playback, and in simulate mode UNLESS a live session is up
  // (then it retargets the running session instead of posing a static robot)
  const liveDrive = mode === "simulate" && !!live;
  const hidden = !robot || !tipMat || playing || (mode === "simulate" && !liveDrive);

  // The handle going away IS a drag end, as far as everything a drag holds is
  // concerned — see releaseDrag. Written as a cleanup so the one path covers
  // both ways it can happen: `hidden` flipping true, and the unmount.
  useEffect(() => {
    if (hidden) return; // nothing can be dragging: there is no handle
    // `refs` is a fresh object each render over the SAME refs, so it carries no
    // state and is deliberately not a dep; `controls` is one, so a swapped
    // OrbitControls is never left disabled by a drag on the old one.
    return () => releaseDrag(refs, controls);
  }, [hidden, controls]);

  if (hidden) return null;

  const send = (m: number[], snap: boolean) => {
    if (liveDrive) void driveTipLive(m);
    else void solveIkGoverned(m, snap);
  };

  const queue = (w: THREE.Matrix4) => {
    if (raf.current) return;
    const world = w.clone();
    raf.current = requestAnimationFrame(() => {
      raf.current = 0;
      // URDF-world target = DISPLAY_UP⁻¹ · (three-world gizmo matrix)
      tmp.copy(DISPLAY_UP_INV).multiply(world);
      send(tmp.toArray(), false); // damped live-follow, no snap
    });
  };

  return (
    <group matrixAutoUpdate={false} matrix={groupMatrix}>
      <PivotControls
        matrix={liveDrive && dragging.current ? (held.current ?? pivotMatrix) : pivotMatrix}
        autoTransform={false}
        disableScaling
        depthTest={false}
        scale={0.18}
        lineWidth={2.5}
        axisColors={["#ff5a5a", "#5aff7a", "#5a9bff"]}
        onDragStart={() => {
          if (controls) controls.enabled = false;
          dragging.current = true;
          held.current = pivotMatrix;
        }}
        onDrag={(_l, _dl, w) => {
          lastWorld.current.copy(w);
          queue(w);
        }}
        onDragEnd={() => {
          releaseDrag(refs, controls);
          // exact final target from the last drag world matrix (same path as
          // queue); off-line that also snaps IK, live it is just the last word
          tmp.copy(DISPLAY_UP_INV).multiply(lastWorld.current);
          send(tmp.toArray(), true);
        }}
      />
    </group>
  );
}

/** The mutable half of a drag, as refs. Grouped so the release below is a plain
 *  function — the component's own `useRef`s satisfy it as they are. */
export interface DragRefs {
  /** queued drag-follow rAF handle, 0 when none is pending */
  raf: { current: number };
  /** a PivotControls drag is in progress (onDragStart → onDragEnd) */
  dragging: { current: boolean };
  /** the handle pose frozen for the duration of a live drag */
  held: { current: THREE.Matrix4 | null };
}

/** Undo everything `onDragStart` did, whether or not the drag ended normally.
 *
 *  OrbitControls is disabled for the duration of a drag, so SOMETHING has to
 *  re-enable it — and `onDragEnd` is not that something on its own: the handle
 *  can be taken off screen mid-drag (playback starts, the live session ends,
 *  the mode changes) and PivotControls then unmounts without ever firing it,
 *  leaving the camera dead for the rest of the app's life and the drag refs
 *  claiming a drag that no longer exists.
 *
 *  `controls.enabled` is only touched while a drag of OURS is up: otherwise
 *  whoever disabled it is not us. Exported for the unit test. */
export function releaseDrag(refs: DragRefs, controls: { enabled: boolean } | undefined): void {
  if (refs.raf.current) {
    cancelAnimationFrame(refs.raf.current);
    refs.raf.current = 0;
  }
  if (!refs.dragging.current) return;
  if (controls) controls.enabled = true;
  refs.dragging.current = false;
  refs.held.current = null;
}
