# Architecture

Caliper is a Cargo workspace of small, focused crates. The `caliper` umbrella
crate re-exports the engine modules; the three faces build on top of it (and, in
a few places, depend on individual sub-crates directly for types the facade does
not re-export).

## Crate map

| Crate | Role |
|------|------|
| `caliper-spatial` | SE(3)/SO(3) screw math — twists, `exp6`/`log6`, adjoints, spatial inertia. Twist ordering is `[v; ω]`, Pinocchio-compatible. |
| `caliper-model` | URDF parsing → a **frozen** struct-of-arrays kinematic `Model`. |
| `caliper-kinematics` | Forward kinematics, geometric Jacobians (world/body), singularity analysis. |
| `caliper-ik` | Inverse kinematics — damped-least-squares / Levenberg–Marquardt CLIK, plus an analytic 6R solver. |
| `caliper-dynamics` | RNEA (inverse dynamics), CRBA (mass matrix), forward dynamics, and a semi-implicit-Euler `Simulator`. |
| `caliper-motion` | Jerk-limited S-curve trajectories (MOVE_J/L/C), waypoint retiming, and a time-optimal (TOPP) parameterization. |
| `caliper-planning` | RRT-Connect / RRT\* / PRM planners, shortcut smoothing, reachability analysis. |
| `caliper-collision` | Self-contained, pure-nalgebra collision checker (OBB-SAT, GJK, EPA, half-space, capsule, mesh-as-hull). |
| `caliper-trajopt` | CHOMP-style collision-aware trajectory optimization over an initial waypoint path. |
| `caliper-hal` | Hardware/sim abstraction: the `RobotBackend` contract, computed-torque control loop, `SafetyMonitor`, teleop, LeRobot dataset record/replay, feature-gated CAN / Dynamixel skeletons. |
| `caliper-calib` | Kinematic (joint-offset / zero) calibration by damped Gauss–Newton. |
| `caliper-graph` | Phase-8 dataflow IR (serde) + deterministic graph executor. |
| `caliper` | Umbrella facade re-exporting the engine modules. |
| `caliper-cli` | The command-line face. |
| `caliper-py` | The Python face (PyO3 / maturin, `import caliper`). |
| `apps/studio` | *Caliper Studio* — the Tauri + React desktop face. |
| `learn/` | The Phase-7 pure-PyTorch behavior-cloning sidecar (`caliper_learn`), a Python package outside the Cargo workspace. |

The umbrella crate's re-exports (`caliper::spatial`, `caliper::kinematics`,
`caliper::ik`, `caliper::dynamics`, `caliper::motion`, `caliper::planning`,
`caliper::collision`, `caliper::trajopt`, `caliper::hal`, `caliper::calib`,
`caliper::graph`, `caliper::model`) map one-to-one onto the crates above.

`apps/studio` is excluded from the default workspace build (it needs a built
frontend); build it with `npm run tauri dev` from `apps/studio`.

## Design principles

- **Lean dependencies.** The engine is `nalgebra` + `std` in spirit. Collision
  is *pure nalgebra* on purpose — `parry`/`rapier` were rejected. Planning uses
  a hand-rolled seeded PRNG rather than pulling in `rand`. The graph executor
  adds only `serde` on top of the engine crates.
- **Determinism / clock-free.** No `Instant::now`, no wall clock anywhere in the
  engine. Time is `t == tick * dt`; simulation and control advance only on an
  explicit `step(dt)`. Randomized planners take a seed. This is what makes the
  whole stack bit-for-bit reproducible and testable without hardware.
- **Frozen model.** `caliper-model` parses a URDF once into an immutable
  struct-of-arrays `Model` that the hot paths (FK, Jacobians, RNEA, CRBA) read
  without re-parsing or re-allocating.
- **No math in the faces.** The CLI, Python bindings, and Studio backend are
  thin: they parse/marshal and dispatch to the engine. The dataflow graph's
  COMPUTE nodes each dispatch to an *existing* engine function — no new math
  lives in `caliper-graph`.
- **Consistent conventions.** Twists and spatial quantities are `[v; ω]` and
  Pinocchio-compatible; the geometric Jacobian comes in a world (LWA-style) and
  a body (LOCAL) flavor, matching the oracle's Pinocchio reference frames.
