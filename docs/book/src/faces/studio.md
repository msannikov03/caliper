# Studio (desktop app)

*Caliper Studio* is the desktop face: a **Tauri** (Rust backend) + **React**
(frontend, with react-three-fiber for the 3D scene, @xyflow/react for the node
graph, and uplot for scopes/plots) application. Five modes share one
persistent 3D canvas (⌘1…⌘5, or the ⌘K command palette):

- **Jog** — live FK, per-joint sliders, IK tip gizmo, singularity HUD +
  manipulability ellipsoid.
- **Motion** — jerk-limited MOVE_J / MOVE_L planning, named poses, playback
  transport.
- **Simulate** — gravity drop, computed-torque drive-to-goal, RRT plan,
  collision check, dynamics readout; MuJoCo contact sim in `--features
  mujoco` builds ([contact simulation](../capabilities/contact-sim.md)).
- **Graph** — the Simulink-style dataflow editor backed by the
  [dataflow graph](../capabilities/studio-graph.md) (run/validate,
  save/load, file import/export, live scopes).
- **Data** — a LeRobotDataset v3.0 browser/editor (episode table, per-channel
  plots, camera thumbnails, tags, delete/split/merge) — reachable with no
  robot loaded.

On first launch a **six-step tour** points out the mode tabs, Open URDF… and
⌘K. It is a pure frontend overlay: skippable at every step, it never blocks
input, never touches the store or session resume, and never shows again once
dismissed or finished (the `caliper.tourDone` localStorage flag). Replay it
any time via ⌘K → *Show tour*.

## Doctors

Both diagnostic engines are wired in (see [Doctors & trajectory
lint](../capabilities/doctors.md)):

- **Asset doctor** — every robot load is diagnosed in the background. When a
  load *fails*, or succeeds with Error-severity findings (e.g. a silently
  dropped collision mesh), the findings appear in the error-banner area with
  severity chips. If any finding is mechanically fixable, a **Repair &
  reload** button runs the repair, writes a sibling `<stem>.repaired.urdf`
  (the input file is never modified), and loads that copy — the HUD then
  labels the session as running on a repaired copy.
- **Dataset doctor** — the **Doctor** button in Data mode streams the open
  dataset through every `D001`–`D015` check and lists the findings; a finding
  that names an episode is clickable and jumps the episode table to it, so a
  bad take can be split or deleted on the spot. Structural edits clear the
  report (it described the pre-edit bytes).

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
