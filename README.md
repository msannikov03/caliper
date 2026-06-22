# Caliper

A modern, open-source robotics engine in Rust — kinematics, IK, singularity
analysis, motion, and real + simulated robot control behind one interface.

**Three faces:** a Tauri desktop app (Caliper Studio), a CLI, and Python
bindings (`import caliper`). One engine, scriptable like MATLAB, controllable
in a polished GUI.

> Phase 0 (scaffold). Core crates compile; faces + math land in later phases.

## Workspace
- `crates/caliper-spatial` — SE(3)/SO(3) math
- `crates/caliper-model` — robot model + URDF loading
- `crates/caliper-kinematics` — FK, Jacobians, singularity
- `crates/caliper-dynamics` — dynamics + trajectories
- `crates/caliper-ik` — inverse kinematics
- `crates/caliper-hal` — `RobotBackend` trait + `SimBackend`
- `crates/caliper` — umbrella facade
