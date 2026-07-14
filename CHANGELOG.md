# Changelog

All notable changes to Caliper are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
the pre-1.0 policy in the
[stability contract](docs/book/src/reference/stability.md): patch releases
never break, minor releases may break **only** with an entry in this file.

Early history is backfilled coarsely (the project moved in phases and waves,
not releases); from `0.1.0` on, every release gets a proper entry.

## [Unreleased]

### Added
- Native **LeRobotDataset v3.0** writer + reader (`caliper-dataset`), exposed
  on all faces (`RecorderV3`/`DatasetReaderV3`, `caliper record`/`replay` with
  auto-detection); our recordings load directly in lerobot 0.4.4.
- Dataset **edit ops** — delete/split/merge episodes + a tags sidecar — and a
  Studio **Data mode** (episode table, channel plots, thumbnails, edit bar).
- **Policy runner**: deploy lerobot ACT hub checkpoints inside caliper's
  control loop, safetensors-only, offline, with the safety monitor live.
- **MuJoCo contact simulation** (optional `mujoco` feature, off by default):
  `caliper-sim-mujoco` crate, MJCF export (`caliper mjcf`, hull-mesh assets),
  a HAL backend so the existing control stack drives contact sim unchanged,
  Studio contact-sim mode with free props, and the dylib bundled in the app.
- **Image features** in datasets (lerobot-exact arrow layout), a headless
  sim-camera collector (offscreen deterministic renders → v3.0 image
  datasets), and a **vectorized sim env** (`VecSimEnv`, gymnasium-vector
  semantics without gymnasium).
- **Doctors**: URDF asset doctor (A-codes, opt-in repair incl.
  inertia-from-mesh), dataset doctor (D-codes, 15 pre-train diagnostics), and
  a trajectory linter (T-codes) — on CLI (`caliper doctor`, `caliper data
  doctor`, `report --strict`), Python, and Studio.
- **Verdicts** for trained policies: seeded eval harness (E-codes, Wilson-95
  aggregates), latency profiler (L-codes, chunk-aware p95), policy autopsy
  (P-codes incl. normalization/cadence mismatch), and a `caliper-learn`
  console script with `--json`.
- OLP-style cycle-time + path-quality **`caliper report`** (`--json`,
  `--strict`) on all three faces.
- Vendored minimal **xacro expander** (pure Rust subset; unsupported
  constructs fail loudly) — `.xacro` files load directly.
- Unified **pose convention** on the Python face (4×4 column-major everywhere)
  and lerobot/robomimic interop exporters.
- Release engineering: GitHub Pages docs deploy, tagged-release wheels
  (macOS arm64 + manylinux x86_64), version-consistency gate.
- Benchmark harness `scripts/measure_lightweight.sh` (+ tests) and three new
  book pages: *Lightweight, measured*, *Stability contract*, *Headless CI
  recipe*. This changelog.
- **Robot zoo**: `caliper fetch <name>` / `--list` materializes a vendored real
  URDF (Panda / SO-101 / SO-100 / Gen3 lite) with license attribution; each
  entry's expected doctor-error set is documented and test-pinned.
- **Studio first-run tour** (dismissible 6-step overlay + palette "Show tour")
  and a *Zero to moving in 10 minutes* quickstart with commands verified
  against the real surfaces.
- **Data factory** (`caliper_learn`): domain randomization (`RandomizationSpec`,
  CI-diffable seeded draws, `VecSimEnv(randomization=)`), a coverage generator
  closing the dataset-doctor → generator loop (`caliper-learn coverage`), and
  MP4 **video** dataset features (`caliper_learn.video`, lerobot-exact encode,
  real round-trip).
- **Contact materials** (`Rigid`/`Rubber`/`Foam`/`Steel`/`Wood`/`Custom`
  presets), a contact **stability linter** (`C001`–`C003` with concrete fixes),
  and a convex-decomposition seam (identity impl) — `caliper-sim-mujoco`.
- Build-program audit page and Data-factory chapter; capability matrix updated
  for the zoo, materials, randomization, coverage, and video.

### Changed
- `caliper record` default dataset format is **v3.0** (was v2.1); the legacy
  layout stays reachable via `--format v21`.
- Python pose-accepting APIs now take the unified 4×4 column-major convention
  (flat-12 row-major grandfathered only in `plan_to_pose`).

### Fixed
- 1-ulp overshoot in motion `sample_grid` (last knot now lands exactly on the
  profile duration).
- MOVE_C long-way-arc geometry (a >360° sweep is now impossible by
  construction) — shipped in 0.1.0, noted here for visibility.

## [0.1.0] — 2026-07-02

First public release: the entire phase build (0–8) plus the hardening waves,
shipped as a signed `.dmg` (Caliper Studio), a CLI, and a Python package built
from one Rust engine.

### Added
- **Phases 0–2**: URDF → frozen model, FK, geometric Jacobians, SE(3) screw
  math; DLS/LM IK + closed-form analytic 6R IK; singularity analysis.
- **Phase 3**: jerk-limited S-curve motion (MOVE_J/L/C), waypoint retiming,
  time-optimal (corner-stop TOPP) parameterization.
- **Phase 4**: RNEA, CRBA, forward dynamics, semi-implicit-Euler simulator —
  cross-validated against Pinocchio (residuals ≈ 1e-9…1e-15).
- **Phase 5**: real-robot HAL (computed-torque control loop, safety monitor,
  teleop), LeRobotDataset v2.1 record/replay, GJK/EPA collision (primitives,
  capsules, STL→convex-hull meshes), CAN/Dynamixel skeletons.
- **Phase 6**: RRT-Connect / RRT\* / PRM planning, shortcut smoothing,
  collision-aware reachability, CHOMP-style trajectory optimization
  (`caliper-trajopt`).
- **Phase 7**: pure-PyTorch behavior-cloning sidecar (`learn/`), zero lerobot
  dependency; deploy primitives on the bindings.
- **Phase 8**: `caliper-graph` dataflow IR + deterministic executor + Studio
  node-graph editor; graph faces on CLI and Python.
- **Post-phase waves (W1–W9)**: external cross-validation oracles (Ruckig,
  SciPy, NumPy-DLS), proptest fuzz, kinematic calibration (`caliper-calib`),
  vitest headless FE-logic harness, mdBook docs site, criterion benches,
  licensing (Apache-2.0 / CERN-OHL-W / CC-BY).
- **R waves (daily-driver + real robots)**: URDF visuals/meshes/mimic
  rendering in Studio, File→Open + recent files, bounded convex-hull builder
  (mesh-heavy robots load), real-robot URDF corpus cross-validation
  (panda/so101/so100/gen3_lite), ⌘K command palette, logging + panic hook,
  session resume, app icon + "Caliper Studio" identity, lerobot 0.4.4 compat
  contract, first CI-built docs.
- Full-system multi-agent audit + first-principles math re-derivation; all
  face-reachable findings fixed (see `docs/VERIFICATION_REPORT.md`).

[Unreleased]: https://github.com/msannikov03/caliper/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/msannikov03/caliper/releases/tag/v0.1.0
