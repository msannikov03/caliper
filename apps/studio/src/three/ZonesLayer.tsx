// Task target zones — the named regions a `*.caliper-task.json` success
// predicate points at ("the cube ends up in `bin`").
//
// A sibling of PropsLayer under its own Z-up→Y-up DISPLAY_UP group: zone
// centers are given in URDF/engine world (Z-up) exactly like prop poses and
// frame matrices, so the same display rotation applies. Each zone is an
// axis-aligned translucent box with its edges drawn on top, so the region reads
// as a volume without hiding the arm or the prop inside it. Nothing here is
// interactive and nothing here is physical: zones are evaluator-side only —
// they emit no MJCF, so they never collide and never appear in a dataset.
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { useStore } from "../store";
import { DISPLAY_UP } from "../coords";
import { ZONE_RGBA_DEFAULT } from "../sim/task";
import type { TaskZone } from "../sim/task";

function ZoneNode({ zone }: { zone: TaskZone }) {
  const rgba = zone.rgba ?? ZONE_RGBA_DEFAULT;
  const color = useMemo(() => new THREE.Color(rgba[0], rgba[1], rgba[2]), [rgba]);
  // three's BoxGeometry takes FULL extents; a zone ships halves
  const geom = useMemo(
    () => new THREE.BoxGeometry(2 * zone.half[0], 2 * zone.half[1], 2 * zone.half[2]),
    [zone.half],
  );
  // EdgesGeometry, not a wireframe material: a wireframed box also draws every
  // triangle diagonal, which reads as noise at this size
  const edges = useMemo(() => new THREE.EdgesGeometry(geom), [geom]);
  useEffect(() => {
    return () => {
      geom.dispose();
      edges.dispose();
    };
  }, [geom, edges]);
  return (
    <group position={zone.center}>
      {/* depthWrite off so the prop inside the zone stays visible through it */}
      <mesh geometry={geom}>
        <meshStandardMaterial
          color={color}
          transparent
          opacity={rgba[3]}
          depthWrite={false}
          roughness={0.8}
          metalness={0}
        />
      </mesh>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color={color} transparent opacity={Math.min(1, rgba[3] + 0.45)} />
      </lineSegments>
    </group>
  );
}

/** Target zones of the loaded task, or nothing at all without one. Zone names
 *  are unique within a task file (the backend rejects duplicates), so the name
 *  is a stable key. */
export function ZonesLayer() {
  const zones = useStore((s) => s.task?.zones);
  const matrix = useMemo(() => DISPLAY_UP.clone(), []);
  if (!zones || zones.length === 0) return null;
  return (
    <group matrixAutoUpdate={false} matrix={matrix}>
      {zones.map((z) => (
        <ZoneNode key={z.name} zone={z} />
      ))}
    </group>
  );
}
