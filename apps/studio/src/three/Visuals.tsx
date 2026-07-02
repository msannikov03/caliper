// URDF <visual> geometry renderer — the robot's real body.
//
// Each visual hangs under its owning link frame: world = frameMatrix · origin,
// composed on the CPU into one matrixAutoUpdate={false} group per visual
// (mirrors RobotView's FrameNode discipline; matrices are column-major Mat4s
// straight off the engine). Primitives are drawn with three's parametric
// geometries (URDF cylinders/capsules are Z-aligned, three's are Y-aligned —
// a +90° X rotation wrapper bridges the convention, same trick as the joint
// markers). Meshes are fetched once per unique path via the `read_mesh`
// binary IPC command (module-level promise cache) and parsed by extension:
// .stl → STLLoader, .glb/.gltf → GLTFLoader, .dae → ColladaLoader.
//
// This module deliberately does NOT import the store at runtime (type-only
// imports), so its pure helpers are unit-testable headlessly.
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { invoke } from "@tauri-apps/api/core";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { ColladaLoader } from "three/examples/jsm/loaders/ColladaLoader.js";
import type { Mat4, VisualInfo } from "../store";

/** Neutral machined-aluminium tone for visuals without a URDF material
 *  (fits the Instrument palette's cool greys). */
const NEUTRAL = "#9aa0aa";

// ---- pure helpers (unit-tested headlessly) ----

/** World pose of a visual: frame world matrix × shape-local origin.
 *  Both inputs are column-major length-16 arrays (THREE.Matrix4 order). */
export function composeWorld(frame: Mat4, origin: Mat4): THREE.Matrix4 {
  return new THREE.Matrix4().fromArray(frame).multiply(new THREE.Matrix4().fromArray(origin));
}

/** Constructor args for a primitive visual, or null for meshes. `zAligned`
 *  shapes need the +90° X wrapper (URDF Z-axis → three Y-axis convention). */
export type PrimitiveSpec =
  | { shape: "box"; args: [number, number, number]; zAligned: false }
  | { shape: "sphere"; args: [number, number, number]; zAligned: false }
  | { shape: "cylinder"; args: [number, number, number, number]; zAligned: true }
  | { shape: "capsule"; args: [number, number, number, number]; zAligned: true };

export function primitiveSpec(v: VisualInfo): PrimitiveSpec | null {
  switch (v.kind) {
    case "box": {
      const h = v.halfExtents ?? [0.01, 0.01, 0.01];
      // three's BoxGeometry takes FULL extents; the engine ships half-extents.
      return { shape: "box", args: [2 * h[0], 2 * h[1], 2 * h[2]], zAligned: false };
    }
    case "sphere":
      return { shape: "sphere", args: [v.radius ?? 0.01, 32, 24], zAligned: false };
    case "cylinder":
      return {
        shape: "cylinder",
        args: [v.radius ?? 0.01, v.radius ?? 0.01, v.length ?? 0.01, 32],
        zAligned: true,
      };
    case "capsule":
      // three's CapsuleGeometry `length` is the core segment — same as URDF.
      return { shape: "capsule", args: [v.radius ?? 0.01, v.length ?? 0.01, 8, 24], zAligned: true };
    case "mesh":
      return null;
    default: {
      const exhaustive: never = v.kind;
      return exhaustive;
    }
  }
}

// ---- mesh loading (one fetch+parse per unique path, promise-cached) ----

type LoadedMesh = THREE.BufferGeometry | THREE.Object3D;

const meshCache = new Map<string, Promise<LoadedMesh>>();
const warned = new Set<string>();

function warnOnce(msg: string) {
  if (warned.has(msg)) return;
  warned.add(msg);
  console.warn(msg);
}

/** Dispose every cached mesh (GPU buffers + materials) and forget the cache.
 *  Called on robot switch / unmount so stale robots don't pin GPU memory. */
export function disposeMeshCache() {
  for (const p of meshCache.values()) {
    p.then(disposeLoaded).catch(() => {});
  }
  meshCache.clear();
  warned.clear();
}

function disposeLoaded(g: LoadedMesh) {
  if (g instanceof THREE.BufferGeometry) {
    g.dispose();
    return;
  }
  g.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.isMesh) {
      mesh.geometry.dispose();
      const m = mesh.material;
      if (Array.isArray(m)) m.forEach((x) => x.dispose());
      else m.dispose();
    }
  });
}

function loadMesh(path: string): Promise<LoadedMesh> {
  let p = meshCache.get(path);
  if (!p) {
    p = fetchAndParse(path);
    meshCache.set(path, p);
  }
  return p;
}

async function fetchAndParse(path: string): Promise<LoadedMesh> {
  // read_mesh returns a binary tauri::ipc::Response → invoke resolves to an
  // ArrayBuffer (older bridges may deliver a plain number[]; normalize).
  const raw = await invoke<ArrayBuffer | number[]>("read_mesh", { path });
  const buf: ArrayBuffer = raw instanceof ArrayBuffer ? raw : new Uint8Array(raw).buffer;
  const dot = path.lastIndexOf(".");
  const ext = dot >= 0 ? path.slice(dot + 1).toLowerCase() : "";
  switch (ext) {
    case "stl":
      return new STLLoader().parse(buf);
    case "glb":
    case "gltf":
      return await new Promise<THREE.Object3D>((resolve, reject) => {
        new GLTFLoader().parse(
          buf,
          "",
          (gltf) => resolve(gltf.scene),
          (e) => reject(e instanceof Error ? e : new Error(String(e))),
        );
      });
    case "dae":
      return new ColladaLoader().parse(new TextDecoder().decode(buf), "").scene;
    default:
      throw new Error(`unsupported mesh format ".${ext}"`);
  }
}

// ---- components ----

/** URDF rgba when present, else the neutral machined tone; alpha<1 → transparent. */
function VisualMaterial({ rgba }: { rgba: VisualInfo["color"] }) {
  const color = useMemo(
    () => (rgba ? new THREE.Color(rgba[0], rgba[1], rgba[2]) : new THREE.Color(NEUTRAL)),
    [rgba],
  );
  const alpha = rgba ? rgba[3] : 1;
  return (
    <meshStandardMaterial
      color={color}
      metalness={0.35}
      roughness={0.5}
      transparent={alpha < 1}
      opacity={alpha}
    />
  );
}

function PrimitiveVisual({ spec, rgba }: { spec: PrimitiveSpec; rgba: VisualInfo["color"] }) {
  return (
    <group rotation={spec.zAligned ? [Math.PI / 2, 0, 0] : [0, 0, 0]}>
      <mesh>
        {spec.shape === "box" && <boxGeometry args={spec.args} />}
        {spec.shape === "sphere" && <sphereGeometry args={spec.args} />}
        {spec.shape === "cylinder" && <cylinderGeometry args={spec.args} />}
        {spec.shape === "capsule" && <capsuleGeometry args={spec.args} />}
        <VisualMaterial rgba={rgba} />
      </mesh>
    </group>
  );
}

/** One visual: a group whose LOCAL matrix is frameMatrix · origin (both from
 *  the engine, column-major), holding either a primitive or a loaded mesh. */
function VisualNode({ v, frame }: { v: VisualInfo; frame: Mat4 | undefined }) {
  const ref = useRef<THREE.Group>(null!);
  const [loaded, setLoaded] = useState<LoadedMesh | null>(null);

  useLayoutEffect(() => {
    if (!ref.current || !frame || frame.length !== 16 || v.origin.length !== 16) return;
    ref.current.matrix.copy(composeWorld(frame, v.origin));
    ref.current.matrixWorldNeedsUpdate = true;
  }, [frame, v.origin]);

  useEffect(() => {
    if (v.kind !== "mesh") return;
    if (!v.meshPath) {
      // unresolvable at parse time — keep the slot empty (rod fallback logic is
      // per-robot, but the triad overlay still shows the frame).
      warnOnce(`caliper: visual mesh unresolved, skipping render: ${v.raw ?? "?"}`);
      return;
    }
    let alive = true;
    const path = v.meshPath;
    loadMesh(path)
      .then((g) => {
        if (alive) setLoaded(g);
      })
      .catch((e) => {
        warnOnce(`caliper: visual mesh failed to load/parse: ${path}: ${String(e)}`);
        if (alive) setLoaded(null);
      });
    return () => {
      alive = false;
    };
  }, [v]);

  // gltf/dae scenes can only have one parent — clone per mount (geometry and
  // materials stay SHARED with the cache original, disposed centrally).
  const instance = useMemo(
    () => (loaded && !(loaded instanceof THREE.BufferGeometry) ? loaded.clone(true) : null),
    [loaded],
  );

  const spec = primitiveSpec(v);
  const scale = v.meshScale ?? ([1, 1, 1] as [number, number, number]);
  return (
    <group ref={ref} matrixAutoUpdate={false}>
      {spec && <PrimitiveVisual spec={spec} rgba={v.color} />}
      {v.kind === "mesh" && loaded instanceof THREE.BufferGeometry && (
        <group scale={scale}>
          {/* geometry passed by prop (not JSX-created) → r3f does NOT auto-
              dispose it on unmount; the module cache owns its lifetime. */}
          <mesh geometry={loaded}>
            <VisualMaterial rgba={v.color} />
          </mesh>
        </group>
      )}
      {v.kind === "mesh" && instance && (
        <group scale={scale}>
          <primitive object={instance} />
        </group>
      )}
    </group>
  );
}

export interface VisualsProps {
  visuals: VisualInfo[];
  /** live per-frame world matrices — the SAME store array RobotView consumes. */
  frames: Mat4[];
  /** identity of the loaded robot; changing it flushes the mesh cache. */
  robotKey: string;
}

/** The robot body: every URDF `<visual>` under its live frame pose. */
export function Visuals({ visuals, frames, robotKey }: VisualsProps) {
  // Flush (dispose) cached meshes whenever the robot changes or the layer
  // unmounts. React runs this cleanup BEFORE the new robot's load effects,
  // so a fresh robot never has its just-cached meshes disposed underneath it.
  useEffect(() => () => disposeMeshCache(), [robotKey]);
  return (
    <group>
      {visuals.map((v, i) => (
        <VisualNode key={`${robotKey}:${i}`} v={v} frame={frames[v.frame]} />
      ))}
    </group>
  );
}
