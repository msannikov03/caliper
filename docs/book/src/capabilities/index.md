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
- [Studio dataflow graph](./studio-graph.md) — the Phase-8 serde IR and
  deterministic executor.
- [Learning sidecar](./learning.md) — the pure-PyTorch behavior-cloning package.
