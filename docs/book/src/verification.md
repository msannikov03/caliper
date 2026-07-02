# Verification

Caliper is built "verify as you go." This chapter is the honest trust map. The
authoritative snapshot lives in
[`docs/VERIFICATION_REPORT.md`](https://github.com/msannikov03/caliper/blob/main/docs/VERIFICATION_REPORT.md);
this page summarizes it and points at the caveats.

## The short version

- **Engine math: independently verified correct.** Ten of eleven algorithm
  clusters were re-derived from first principles and matched line-by-line at high
  confidence; the eleventh (numerical-stress) raised robustness/coverage items,
  not math errors.
- **Headless stack (engine + CLI + PyO3): machine-verified and trustworthy.**
- **Studio GUI: built, build-checked, statically reviewed — but never
  executed.** Runtime behavior is unverified, by deliberate choice.

## Four independent lines of evidence

### 1. External cross-validation (Pinocchio + NumPy + Ruckig-class + SciPy)

A Python oracle (`oracle/`) runs **through the PyO3 bindings** and compares
against reference implementations:

- **Pinocchio** — FK, geometric Jacobian (world = LWA, body = LOCAL), RNEA,
  CRBA, and forward dynamics, with residuals ≈ **1e-9…1e-15**.
- **NumPy SVD** — singularity metrics (σ, manipulability, condition number).
- **Ruckig-class expectations** — motion profiles are sanity-checked against
  jerk-limited expectations (there is no Ruckig oracle wired in — see the caveat
  below).
- **LeRobot schema** — dataset record/replay validated against the schema via
  pyarrow + NumPy stats (lerobot itself is not importable in the test env).
- **SciPy/NumPy** — control checked where a closed form exists (e.g. the 2-DOF
  computed-torque case).

Because the oracle goes through the shipped bindings, it validates the Python
face and the shared core simultaneously. This external check has caught real
bugs — an RNEA sign error was found exactly this way.

### 2. Property tests

Proptest-style invariants on the math: round-trips, monotonicity,
endpoint-exactness, and limit adherence.

### 3. First-principles re-derivation + multi-agent audit

An independent re-derivation of all eleven algorithm clusters (line-by-line, at
high confidence) plus a large multi-agent correctness/safety audit. Every
confirmed finding was fixed or explicitly documented. The most recent pass
recorded 13 findings, all addressed (Cartesian move validation symmetry, a
`Simulator::step` non-positive-`dt` guard, a 0-row-Jacobian manipulability guard,
and a documented Earth-gravity scope note, among others).

### 4. Studio FE-logic harness (vitest)

The parts of the Studio frontend that *can* be tested off-screen are: a **vitest**
harness covers coordinate transforms (`coords.test.ts`), the app store
(`store.test.ts`), and graph serialize/deserialize (`graph/serialize.test.ts`).
This tests logic, **not** rendering.

## The honest gaps

These are stated plainly because pretending otherwise would be the real defect:

1. **The GUI has never been run.** The entire Studio (all its modes plus the node
   editor) is build-checked and statically reviewed only. Its rendering and
   interactions are unverified. The first `tauri dev` is the real test.
2. **Self-consistent-only clusters.** Several components are re-derived-correct
   but validated only against Caliper itself, not an independent reference:
   SE(3) `log6`/`V⁻¹` and its small-angle branch; the adjoint and 6×6 spatial
   inertia; the IK **solver** (validated via FK∘IK closure, not a task-space DLS
   reference); the manipulability-ellipsoid eigendecomposition; the redundant-arm
   nullspace; **all of `caliper-motion`** (no Ruckig-class oracle); RRT/smoothing;
   OBB-SAT; the computed-torque decoupling; and the LeRobot dataset (schema, not
   lerobot). A defect *shared* between a forward and inverse path (e.g. a
   compensating error in both `exp` and `log`) would not be caught by closure
   tests — low risk given the re-derivation, but not machine-caught.
3. **By-design limitations (documented, not bugs).** The collision guarantee is
   sampled-at-resolution (narrow passages can tunnel); mesh/capsule colliders that
   can't be reduced to supported primitives are surfaced via `uncovered_frames`
   rather than checked; the native `Simulator` has no collision; the "singular
   joint" classification is advisory. (`MOVE_C`, formerly unwired dead code, is
   now fixed — short-way arc through the via — wired to the CLI/Python faces and
   oracle-covered.) There are no dedicated narrow-passage / near-π / at-limit stress
   fixtures yet — coverage is random sampling.

## Repro pointers

```sh
just ci        # fmt-check + clippy + test + lean-core check
just test      # cargo test --workspace --exclude studio
just oracle    # Pinocchio/NumPy cross-validation (needs the repo .venv)
just learn     # pure-PyTorch BC sidecar tests
```

The oracle and learning tests need a Python venv with `maturin`, `pinocchio`,
`numpy`, `pyarrow` (and `torch` for `learn`). See the `justfile` recipe comments.
