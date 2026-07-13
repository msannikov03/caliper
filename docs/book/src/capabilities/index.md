# Capabilities

This section is a per-capability guide to the engine. Each page describes what
the algorithm does, the conventions it follows, and — importantly — how far it
has actually been verified. Cross-validated numbers, self-consistent-only
components, and by-design limitations are all called out explicitly; the whole
trust map is collected in the [verification chapter](../verification.md).

- [Kinematics & IK](./kinematics.md) — FK, geometric Jacobians, DLS/LM and
  analytic 6R inverse kinematics, singularity analysis.
- [Motion](./motion.md) — jerk-limited S-curve MOVE_J/L/C, waypoint retiming,
  time-optimal (TOPP) parameterization.
- [Dynamics & simulation](./dynamics.md) — RNEA, CRBA, forward dynamics, the
  `Simulator`.
- [Planning](./planning.md) — RRT-Connect / RRT\* / PRM, shortcut smoothing,
  reachability, CHOMP-style trajectory optimization.
- [Collision](./collision.md) — OBB-SAT, GJK, EPA penetration depth, half-space,
  capsule, mesh-as-convex-hull.
- [Control & safety](./control-safety.md) — the backend contract, the
  computed-torque control loop, the safety monitor, teleop, dataset record.
- [Calibration](./calibration.md) — joint-offset (zero) calibration.
- [Doctors & trajectory lint](./doctors.md) — the asset doctor
  (`A001`–`A014`, with mechanical repair), the dataset doctor (`D001`–`D015`),
  and the trajectory lint (`T001`–`T009`).
- [Studio dataflow graph](./studio-graph.md) — the Phase-8 serde IR and
  deterministic executor.
- [Learning sidecar](./learning.md) — the pure-PyTorch behavior-cloning package.
- [Verdicts — eval, profiling & the Policy Autopsy](./verdicts.md) — the
  seeded eval harness (`E001`–`E003`), the deploy-loop latency profiler
  (`L001`–`L003`), the policy debugger (`P001`–`P008`), and the autopsy that
  merges them under one verdict.

For a single table of every capability against the face(s) that expose it —
including honest gaps — see the [capability
matrix](../reference/capability-matrix.md).
