# Caliper

**A modern, open robotics engine — one deterministic Rust core, three faces.**

Caliper is a single Rust engine for serial-arm robotics: kinematics, inverse
kinematics, singularity analysis, jerk-limited motion, dynamics and simulation,
collision-aware planning, real-robot control and safety, kinematic calibration,
and a Simulink-style dataflow graph. The same engine code is exposed through
three faces:

- **CLI** — `caliper fk | ik | analyze | move | plan | sim | record | graph …`
- **Python** — `import caliper`, built with [maturin](https://www.maturin.rs/) /
  PyO3; scriptable like MATLAB/NumPy.
- **Studio** — *Caliper Studio*, a Tauri + React desktop app with a 3D scene and
  a node-graph editor.

## One engine, three faces

There is exactly one implementation of every algorithm. The CLI parses
arguments and calls the engine; the Python bindings marshal NumPy arrays and
call the engine; Studio serializes a dataflow document and calls the engine.
None of the faces re-implement math. A consequence worth stating up front: when
the Python oracle validates FK against Pinocchio, it is validating the shipped
Python face *and* the shared core at once, because the oracle runs *through* the
PyO3 bindings.

The engine is **deterministic and clock-free** by design. Nothing consults the
wall clock; simulation and control advance only when a `step(dt)` is called, and
the one randomized component (RRT/RRT\* sampling) uses a seeded splitmix64 PRNG
rather than `rand`. A given input — including a given seed — produces the same
output every time, which is what makes the whole stack unit-testable with no
hardware.

## Status (honest)

All nine phases of the build (0–8) exist and compile:

| Phase | Capability |
|------|------------|
| 0–1 | URDF → frozen kinematic model, forward kinematics, geometric Jacobians, SE(3)/SO(3) screw math |
| 2 | DLS/LM inverse kinematics + analytic 6R IK; singularity analysis |
| 3 | Jerk-limited S-curve motion (MOVE_J/L/C) + waypoint retiming + time-optimal (TOPP) parameterization |
| 4 | Inverse dynamics (RNEA), mass matrix (CRBA), forward dynamics, a semi-implicit-Euler `Simulator` |
| 5 | Real-robot backend contract, computed-torque control loop, safety monitor, teleop, LeRobot dataset record/replay |
| 6 | RRT-Connect / RRT\* / PRM planning, shortcut smoothing, reachability, CHOMP-style trajectory optimization |
| 7 | A pure-PyTorch behavior-cloning sidecar (`learn/`) |
| 8 | A serde dataflow IR + deterministic graph executor + a node-editor face |

**What is trustworthy vs. what is not:**

- The **headless stack** (engine + CLI + PyO3 bindings) is machine-verified:
  cross-validated against Pinocchio and NumPy (residuals ≈ 1e-9…1e-15), covered
  by ~156 Rust tests and a Python oracle, and put through an independent
  first-principles re-derivation plus a large multi-agent correctness/safety
  audit.
- The **Studio GUI** compiles, type-checks, builds, and was statically
  reviewed — **but it has never been launched at runtime.** Its rendering and
  interactions are *not* verified. This is deliberate (build now, human-review
  later), and it is called out honestly throughout this book and in the
  [verification chapter](./verification.md).

This documentation describes only what is actually implemented. Where something
is a stub, a by-design limitation, or unverified, it says so.
