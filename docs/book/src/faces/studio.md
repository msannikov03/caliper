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
  mujoco` builds ([contact simulation](../capabilities/contact-sim.md)); a
  **live session** (Start Live / Pause / Reset / Stop) stepping the sim in
  real time — see below.
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

## Live session (Simulate)

Alongside the baked rollouts, Simulate mode runs a **live stepped session**:

- **Start Live** spawns a background thread that steps the sim at a fixed
  1 ms physics timestep with a PD servo holding a live-mutable joint target,
  and streams state to the viewport at render rate (~60 Hz). In `mujoco`
  builds this is the full contact sim — free props supported, contact count
  shown live; default builds fall back to the builtin gravity integrator
  (no contacts, props rejected with a clear error).
- **Drive it by hand** — while live, the joint sliders edit the PD hold
  target (the measured pose rides along as a ghost tick so the servo lag is
  visible), the IK gizmo drags the tip, `[`/`]` pick a joint and `-`/`=` or
  the arrow keys jog it, and a gamepad drives the tip in cartesian (A pauses,
  B resets). Space freezes/unfreezes.
- **Pause** freezes the sim — stepping and the wall clock both stop, and the
  arm holds its pose. It is a freeze, not an e-stop: nothing is de-energized,
  so nothing falls.
- **Reset** returns deterministically to the start pose (on MuJoCo, a full
  `mj_resetData` including the warmstart) with the session clock back at
  zero; it works while paused and updates the viewport immediately.
- **Stop** ends the session. A stepping error also ends it, with the reason
  surfaced rather than a silent freeze.

- **Grasp props** — robots with a gripper joint (auto-detected by name, or
  named explicitly) get a gripper open/close control (button, `G`, or gamepad
  X); closing on a touching prop welds it to the gripper — the standard sim
  teleop heuristic, labeled as such — and a `HELD` badge names what's carried.
- **Record teleop episodes** — while live, pick a dataset folder, set a task
  label and fps (default 50), and record takes straight into a native
  LeRobotDataset v3.0: stop-and-save or discard per take, episode counter,
  finish-dataset, then open the result in Data mode. Capture is exact tick
  decimation in the sim thread (timestamps are `k/fps`, not wall-clock).
  Reset discards the current take — a reset invalidates the demonstration.

Bake-then-replay stays for what it is good at — reproducible clips and the
`C001`–`C003` stability lint. Live is for watching, driving, and recording
demonstrations. Details and honest constraints:
[Live session](../capabilities/contact-sim.md#live-session-studio).

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
