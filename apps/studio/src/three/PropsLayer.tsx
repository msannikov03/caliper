// Contact-sim free props — render-only playback of the baked per-frame prop
// poses riding on a contact SimTrajectoryDto (`props: PropTrack[]`).
//
// A sibling of RobotView under its OWN Z-up→Y-up DISPLAY_UP group: prop poses
// come from MuJoCo in URDF/engine world (Z-up), exactly like the robot's frame
// matrices, so the same display rotation applies. Each prop is one group with
// matrixAutoUpdate={false} whose LOCAL matrix is composed from the baked
// [x,y,z,qw,qx,qy,qz] row at the current playback instant (same nearest-frame
// rounding as _applyTrajAt, via frameIndexAt) — mirroring FrameNode's
// discipline. Cylinders are Z-aligned on the wire (MJCF/URDF convention);
// three's are Y-aligned, bridged by the same +90° X wrapper Visuals uses.
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useStore } from "../store";
import { DISPLAY_UP } from "../coords";
import { frameIndexAt } from "../sim/props";
import type { PropTrack } from "../sim/props";

/** Neutral accent for props without an rgba (matches the joint-marker teal). */
const PROP_NEUTRAL = "#36c6d4";

function PropNode({ track, k }: { track: PropTrack; k: number }) {
  const ref = useRef<THREE.Group>(null!);
  const f = track.frames[k];

  useLayoutEffect(() => {
    if (!ref.current || !f || f.length !== 7) return;
    ref.current.matrix.compose(
      new THREE.Vector3(f[0], f[1], f[2]),
      // wire order is w-first (MJCF); THREE.Quaternion takes (x, y, z, w)
      new THREE.Quaternion(f[4], f[5], f[6], f[3]),
      new THREE.Vector3(1, 1, 1),
    );
    ref.current.matrixWorldNeedsUpdate = true;
  }, [f]);

  const color = useMemo(
    () =>
      track.rgba
        ? new THREE.Color(track.rgba[0], track.rgba[1], track.rgba[2])
        : new THREE.Color(PROP_NEUTRAL),
    [track.rgba],
  );
  const alpha = track.rgba ? track.rgba[3] : 1;
  const material = (
    <meshStandardMaterial
      color={color}
      metalness={0.3}
      roughness={0.45}
      transparent={alpha < 1}
      opacity={alpha}
    />
  );
  const h = track.halfExtents ?? [0.01, 0.01, 0.01];
  const r = track.radius ?? 0.01;
  const l = track.length ?? 0.01;
  return (
    <group ref={ref} matrixAutoUpdate={false}>
      {track.kind === "box" && (
        <mesh>
          {/* three's BoxGeometry takes FULL extents; the wire ships halves */}
          <boxGeometry args={[2 * h[0], 2 * h[1], 2 * h[2]]} />
          {material}
        </mesh>
      )}
      {track.kind === "sphere" && (
        <mesh>
          <sphereGeometry args={[r, 32, 24]} />
          {material}
        </mesh>
      )}
      {track.kind === "cylinder" && (
        <group rotation={[Math.PI / 2, 0, 0]}>
          <mesh>
            <cylinderGeometry args={[r, r, l, 32]} />
            {material}
          </mesh>
        </group>
      )}
    </group>
  );
}

/** Free props of the active contact clip, following playback time. Renders
 *  nothing for builtin rollouts / motion clips / no-mujoco builds (a contact
 *  clip is the ONLY source of prop tracks). */
export function PropsLayer() {
  const simTraj = useStore((s) => s.simTraj);
  const playhead = useStore((s) => s.playhead);
  const matrix = useMemo(() => DISPLAY_UP.clone(), []);
  const tracks = simTraj?.kind === "contact" ? (simTraj.props ?? []) : [];
  if (tracks.length === 0) return null;
  const dt = simTraj!.dt;
  return (
    <group matrixAutoUpdate={false} matrix={matrix}>
      {tracks.map(
        (tr) =>
          tr.frames.length > 0 && (
            <PropNode key={tr.name} track={tr} k={frameIndexAt(playhead, dt, tr.frames.length)} />
          ),
      )}
    </group>
  );
}
