# Caliper

A modern, open-source robotics engine in Rust — kinematics, IK, singularity
analysis, motion, and real + simulated robot control behind one interface.

**Three faces:** a Tauri desktop app (Caliper Studio), a CLI, and Python
bindings (`import caliper`). One engine, scriptable like MATLAB, controllable
in a polished GUI.

> Phases 0–7 complete: kinematics/IK/singularity, motion (S-curve MOVE_J/L/C),
> dynamics (RNEA/CRBA/forward-dynamics) + sim, real + simulated robot control,
> RRT-Connect planning + reachability, and a pure-torch learning sidecar
> (`learn/`). All three faces (Studio app, CLI, Python bindings) are wired.

## Workspace
- `crates/caliper-spatial` — SE(3)/SO(3) math
- `crates/caliper-model` — robot model + URDF loading
- `crates/caliper-kinematics` — FK, Jacobians, singularity
- `crates/caliper-dynamics` — dynamics + trajectories
- `crates/caliper-motion` — S-curve trajectories (MOVE_J/L/C) + retiming
- `crates/caliper-ik` — inverse kinematics
- `crates/caliper-collision` — pure-nalgebra collision model
- `crates/caliper-planning` — RRT-Connect planner + reachability
- `crates/caliper-hal` — `RobotBackend` trait + `SimBackend` + real-robot control
- `crates/caliper` — umbrella facade
- `crates/caliper-cli` — command-line face
- `crates/caliper-py` — Python bindings (`import caliper`)
- `apps/studio` — Tauri desktop app (Caliper Studio)
- `learn/` — Phase 7 pure-torch behavior-cloning sidecar (`caliper_learn`)

## Known API inconsistencies (Python face)

The Cartesian-pose entry points do not yet share one convention — be careful
mixing them:

- `Robot.ik(target, seed, frame=None)` / `Robot.move_l(q_start, target, frame=None)`
  take `target` as a **4×4 column-major** matrix (`Vec` of 4 columns, each
  length 4; `target[col][row]`) and `frame` as a **frame NAME** (`str`,
  defaulting to the tip frame).
- `Planner.plan_to_pose(start, target, frame=None)` takes `target` as a
  **flat 12-element** pose (9 row-major rotation entries followed by `tx, ty, tz`)
  and `frame` as a **frame INDEX** (`int`, defaulting to the tip frame).

So both the pose layout (4×4 column-major vs flat-12 row-major) and the frame
selector (name vs index) differ between these calls. This is a documented wart,
not a deliberate design; unifying it is future work.
