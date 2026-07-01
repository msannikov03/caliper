# Studio (desktop app)

*Caliper Studio* is the desktop face: a **Tauri** (Rust backend) + **React**
(frontend, with react-three-fiber for the 3D scene, @xyflow/react for the node
graph, and uplot for scopes) application. It has two modes: a **3D scene** and a
Simulink-style **Graph** tab backed by the [dataflow
graph](../capabilities/studio-graph.md).

## Launch

```sh
cd apps/studio
npm install
env -u CONDA_PREFIX npm run tauri dev
```

## ⚠️ Not runtime-verified

This is the single most important honesty note in the whole project. The Studio
GUI:

- **compiles** (the Tauri Rust backend),
- **type-checks and builds** (the React/TypeScript frontend, `tsc` + `vite`),
- was **statically reviewed**, and
- has **FE-logic covered by a vitest harness** (coordinate transforms, the
  store, graph serialize/deserialize) — see the [verification
  chapter](../verification.md).

But it **has never been launched at runtime.** No human has watched it render.
Its 3D rendering, its interactions, and its live behavior are *unverified by
deliberate choice* (build-fast-now, human-review-later). Treat the first
`tauri dev` as the real test.

The Tauri backend has been hardened defensively (lock/path/NaN guards, safe
lock-release), and the frontend logic that *can* be unit-tested off-screen is
tested — but none of that substitutes for actually running the app.

> Note: the repository's `just app` recipe mirrors the `npm run tauri dev`
> command above. If `just` is not installed in your environment, use the raw
> `npm` command directly.
