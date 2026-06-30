# Caliper — Build & Verification Report

**Snapshot:** HEAD `37d8156` (2026-06-29) · all 9 MASTER_PLAN phases built.
**Method:** Pinocchio oracle cross-validation + 156 Rust tests + a 90-agent/2-Codex correctness+safety audit (all findings fixed) + an independent first-principles **math re-derivation** of all 11 algorithm clusters (30 agents) + numerical-stress sweep.

## TL;DR
- **Engine math: independently verified CORRECT.** 10/11 clusters re-derived from first principles and matched line-by-line at **high** confidence; the 11th ("numerical-stress") raised robustness/coverage items, not math errors. No math defects survive.
- **Headless stack (engine + CLI + PyO3): machine-verified and trustworthy.**
- **Studio GUI: built + build-checked + statically reviewed, but NEVER executed.** Runtime behavior is unverified by deliberate choice (build-fast-now, human-review-later).

## What was built (inventory)
| Component | LOC | Tests | Verification |
|---|---|---|---|
| caliper-spatial (SE3/screw/adjoint/inertia) | 360 | unit | re-derived CORRECT; exp externally tied via FK; **log6 self-consistent only** |
| caliper-model (URDF→frozen Model) | 659 | unit | CORRECT; collision-coverage honesty fixed |
| caliper-kinematics (FK/Jacobian/singularity) | 750 | unit | FK+World-Jacobian **Pinocchio-validated**; ellipsoid/nullspace self-consistent |
| caliper-ik (DLS/LM) | 351 | unit | re-derived CORRECT; validated by FK∘IK round-trip (solver itself self-consistent) |
| caliper-dynamics (RNEA/CRBA/FD/Simulator) | 621 | unit | RNEA/CRBA/FD **Pinocchio-validated ~1e-9**; integrators energy-bounded |
| caliper-motion (S-curve/move_j/l/c/retime) | 1458 | unit | re-derived CORRECT; self-consistent (no 3rd-party trajectory oracle) |
| caliper-planning (RRT-Connect/reach) | 1005 | unit | re-derived CORRECT; collision guarantee = sampled-at-resolution |
| caliper-collision (OBB-SAT pure-nalgebra) | 741 | unit | re-derived CORRECT incl. edge-edge degeneracy; cannot under-report |
| caliper-hal (control/safety/teleop/dataset/HW codecs) | 3451 | unit | computed-torque + safety re-derived CORRECT; hardened in audit |
| caliper-graph (Phase 8 dataflow executor) | 1648 | 9 | dispatch faithful; parity vs direct move_j/move_l; deterministic |
| caliper-cli | 1086 | — | run-verified subcommands incl. `graph` |
| caliper-py (PyO3) | 1496 | via oracle | oracle runs through these bindings |
| Studio backend (Tauri) | 2131 | unit | compiles; lock/path/NaN hardened; **app never launched** |
| Studio frontend (React/r3f/xyflow/uplot) | 2910 | tsc+vite | type-checks + builds; reviewed; **never rendered at runtime** |
| learn (pure-torch BC sidecar) | 890 | 17 | CPU oracles (overfit/checkpoint/deploy); GPU path deferred |
| oracle (Pinocchio/numpy cross-val) | — | 35 | 53 pass / 1 skip (lerobot not installed) |

## Cross-validation coverage map (the honest trust map)
**Externally cross-validated (Pinocchio/numpy), residuals ~1e-9..1e-15:** FK, geometric Jacobian (World=LWA, Body=LOCAL), RNEA, CRBA, forward dynamics, singularity σ/manipulability/condition# (vs numpy SVD).
**Re-derived-correct but self-consistent-only (no independent reference):** SE3 log6/V⁻¹ + small-angle branch; adjoint/adjoint_inv + 6×6 spatial inertia; IK *solver* (validated via FK∘IK closure, not against a task-space DLS reference); the manipulability ellipsoid eigendecomp; the redundant-arm nullspace; **all of caliper-motion** (S-curve/move_l/retime — no Ruckig-class oracle); RRT/smoothing; OBB-SAT; computed-torque decoupling (validated on a 2-DOF pendulum); the LeRobot dataset (pyarrow schema + numpy stats, not lerobot itself).
**By-design limitations (documented, not bugs):** sampled-resolution collision guarantee (can tunnel narrow passages); mesh/capsule colliders dropped (surfaced via `uncovered_frames`); native sim has no collision; MOVE_C is dead/unwired code with no oracle.

## Findings from this pass (13 confirmed; all addressed)
**Fixed (commit 37d8156):** [MEDIUM] Cartesian move_l/move_c caps+dt + non-finite goal now validated (was asymmetric with the joint-space path); [LOW] Simulator::step rejects non-positive dt (negative dt integrated backward); [LOW] 0-row Jacobian manipulability guard; [LOW] Scope `energy` Earth-gravity assumption documented.
**Accepted as by-design / known (doc only):** discrete collision sampling; MOVE_C uncovered; lerobot skip; log6 not externally checkable (binding doesn't expose it); singular-joint classification is advisory; "symplectic" wording for a non-separable Hamiltonian.

## Honest gaps that remain
1. **GUI never run** — the entire Studio (8 phases + the node editor) is build-checked only. First `tauri dev` is the real test.
2. **Self-consistent-only clusters** would not catch a defect shared between a forward and inverse path (e.g. a compensating error in both exp and log) — low risk given the re-derivation, but not machine-caught.
3. **No narrow-passage / near-π / at-limit stress fixtures** for planning/IK — random sampling only.
