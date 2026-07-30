# Changelog

All notable changes to Caliper are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
the pre-1.0 policy in the
[stability contract](docs/book/src/reference/stability.md): patch releases
never break, minor releases may break **only** with an entry in this file.

Early history is backfilled coarsely (the project moved in phases and waves,
not releases); from `0.1.0` on, every release gets a proper entry.

## [Unreleased]

Nothing yet.

## [0.2.1] — 2026-07-30

### Added
- **Auto-update**: Studio checks the signed `latest.json` on the newest
  GitHub release shortly after launch (minisign signature verified against
  the public key baked into the app before anything is trusted) and shows an
  `update vX ⟳` chip in the toolbar; installing is strictly click-to-consent
  and relaunches in place. Offline or unreachable-endpoint checks degrade to
  a log line, never an error.
- **Policy-in-the-loop bridge**: Studio's live session can be driven by a
  trained policy from a python env you point it at ("connect policy…" in the
  Live block). Studio spawns `caliper-learn drive CKPT --urdf …` and speaks
  a pure-JSON stdio protocol (obs out at a configurable rate, default 20 Hz;
  actions in, latest-wins, clamped to joint limits, limitless joints never
  clamped); the sim thread never blocks on the child — a one-slot latest-wins
  observation bus feeds a dedicated driver/reader/stderr-pump thread trio,
  and a wedged or dead python ends the drive loudly while the sim keeps
  running. Human inputs still nudge mid-drive, Space pauses the policy with
  the sim, session end reaps the child (no orphans), and recording during a
  policy drive is deliberately allowed — that's how policy-rollout datasets
  are made. State-based policies only: camera-demanding checkpoints are
  refused at load by name.

### Fixed
- Post-release adversarial review (three independent reviewers over the full
  0.2.0 diff; every finding independently verified before fixing): a
  recording request retried after a >2 s save/finalize could consume the
  timed-out request's reply as its own (requests now carry correlation ids);
  a start/stop race could strand a legitimate-looking empty dataset at the
  chosen root (now reclaimed and removed); an episode-replay bake landing
  after a dataset switch adopted the wrong dataset's clip (now
  identity-fenced); an interrupted live IK drag left orbit controls disabled
  until reload; a double-pressed "start live" could double-start; a stale
  live IK solve could land on a newer session; robot/task/mode changes now
  tear the live session down synchronously instead of waiting for the ended
  event; a partially-failed event subscription could leak a listener; and
  python's `Lifted(ref="absolute").reset` no longer requires the scored prop
  at reset time (cross-face parity with the Rust evaluator, which never did).

## [0.2.0] — 2026-07-30

The human-demonstration-loop release: Studio's Simulate mode grew a live
stepped sim you can drive by hand (sliders, IK gizmo, keyboard, gamepad),
grasp with (an honest weld heuristic), and record from — straight into
native LeRobotDataset v3.0 datasets that load in real lerobot 0.6.0. One
`*.caliper-task.json` describes a manipulation task for every face, with
success predicates implemented twice (Rust + Python) and pinned to parity.
Data mode replays episodes on the 3D robot and renders `caliper-learn`
verdicts. Plus: a convex-hull orientation bug that was silently degrading
48% of real robot collision meshes is fixed, mesh-robot loads are 3–6×
faster, video metadata is native in the Rust writer, and CI watches the
newest lerobot monthly.

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
- **Studio live sim session** (Simulate mode): a background thread steps the
  contact sim (MuJoCo — props + live contact count; the builtin gravity
  integrator in MuJoCo-free builds) at a fixed 1 ms timestep with a PD hold
  target, streaming state to the viewport at ~60 Hz, with Start Live / Pause
  (a freeze, not a de-energize) / deterministic Reset (full MuJoCo reset,
  warmstart included) / Stop. Bake-then-replay stays for reproducible clips
  and the stability lint. First phase of the human-demonstration-loop
  program.
- **Studio live-session input layer**: the live sim is drivable by hand —
  joint sliders edit the PD hold target (measured pose drawn as a ghost tick,
  so servo lag is visible), the IK gizmo retargets the tip live, `[`/`]`
  select a joint and `-`/`=`/arrows jog it, a gamepad drives the tip in
  cartesian (0.15 deadband, cubic response; A pauses, B resets), and Space
  freezes/unfreezes. One target update per rendered frame; inputs write only
  the hold target, never the streamed pose, so input and stream cannot fight.
- **Studio teleop episode recording**: record takes from a live session
  straight into a native LeRobotDataset v3.0 — capture happens in the sim
  thread at exact tick decimation (default 50 fps from the 1 kHz loop), with
  per-episode task labels, stop-and-save / discard, an episode counter,
  finish-dataset, and open-in-Data. Pause freezes capture into the same take;
  reset discards the take (a reset invalidates the demonstration); ending the
  session finalizes the open dataset. Verified end-to-end: a Studio-recorded
  dataset loads in real lerobot 0.6.0. Adds an additive
  `DatasetWriter::discard_buffered` to `caliper-dataset`.
- **Gripper channel + weld attach-on-grasp** (live session): gripper joints
  auto-detect by joint *or child-link* name (mimic joints skipped; explicit
  override supported), open/close rides the same PD hold target as every
  other input (`G`, gamepad X, or the panel button), and closing on a
  touching prop welds it to the attach link with its relative pose captured
  at activation (snap-free, measured < 1 mm) — the standard sim-teleop
  heuristic, labeled as such, not finger physics. Opening releases; reset
  releases and reopens. New `caliper_model::gripper` detection module, MJCF
  inactive-weld emission behind `MjcfOptions::attach_link`, runtime weld
  (de)activation on `MujocoSim`.
- **Success predicates** (`caliper_learn.success`): `Lifted` /
  `PlacedInZone` / `AllOf` / `AnyOf` with an exact JSON round-trip schema;
  `VecSimEnv(success=…)` reports `info["success"]`/`info["final_success"]`
  per the final-observation convention; the eval harness scores
  predicate-based success (wilson intervals unchanged) and the autopsy
  verdict names the criterion. En route, the vectorized env's dof addressing
  became name-resolved — a prop free-joint in `extra_xml` previously landed
  *before* the robot in qpos order, so scene props would have silently
  received the arm's commands; props now also render in image observations.
- **Episode replay in Data mode**: a dataset episode's recorded joint rows
  play back on the docked 3D robot (the take's own positions through the
  loaded robot's FK — velocities are not invented), with an in-panel
  frame-accurate transport; dataset-doctor findings that know an instant
  (`D006`/`D010`/`D011` now carry frame locations) jump the robot straight
  to that pose, landing paused. Refuses honestly on ndof mismatch,
  non-finite rows, or >20k frames — never silently truncates.
- **Verdict viewers in Data mode**: `caliper-learn eval/debug/profile/
  autopsy --json` documents open in Studio — success rate with the Wilson
  interval drawn as a bar, per-episode tables, `E`/`P`/`L` findings with the
  doctor panel's severity vocabulary, and the autopsy's verdict line —
  detected by structural fingerprint, tolerant of additive fields, loud on
  structurally wrong files.
- **The task zoo** (`tasks/`): five graduated, solvability-witnessed starter
  tasks on the bundled gripper arm (lift, place, hold-high, sort, precise
  place) covering every predicate kind and four material presets. Difficulty
  was measured (admissible-pose counts from an FK sweep), not claimed; both
  suites assert no zoo task is born solved and every prop is reachable.
- **Convex hull builder fixed** — the incremental hull oriented seed faces
  against the whole-cloud centroid instead of the seed tetrahedron's, so
  bracket/shell-shaped meshes collapsed and fell back to raw point clouds:
  measured across a six-family robot zoo, **48% of real collision meshes**
  (206/430) were falling back; now 3.3%, with 47% fewer collider points and
  mutation-checked containment/support-equivalence tests. Collision
  semantics are unchanged (a support function over a point set equals its
  hull's); MJCF collider output legitimately shrinks.
- **Native video features in the dataset writer**: `dtype: "video"` is
  first-class — the writer emits the `videos/{key}/*` episode columns,
  `info.json` entry, `video_path`, and pixel stats in one pass, with
  coherence gates (span-vs-frame-count, mp4-exists, no orphaned
  registrations). Python still encodes the MP4s; the pyarrow post-write
  bridge (`attach_video_metadata`) is retired from the writing path and
  demoted to a documented repair tool, with a test proving native and
  bridge output are equal down to every `meta/episodes` row.
- **The task artifact** (`*.caliper-task.json`, version 1, under the
  stability contract): one JSON file — robot, scene props/materials, named
  zones, gripper override, success predicate, horizon, fps — consumed by
  every face. Studio's *Open task…* loads the robot, places the scene, draws
  the zones, pre-fills recording, and streams a live `SUCCESS` verdict;
  `caliper_learn.load_task` / `VecSimEnv.from_task` / `caliper-learn eval
  --task` / `autopsy --task` consume the same file; the success predicates
  are implemented in both Rust and Python with a shared parity table
  (knife-edge rows included) run by both suites. Unknown keys at any level
  are rejected loudly. Along the way: `PropDto`/Studio props gained
  `material`, the Python face's `model_to_mjcf` gained a structured `props=`
  kwarg (one source of truth — the Rust prop path), `VecSimEnv` takes
  `props=`, and `MujocoSim` exposes `prop_velocities()`.
- **Faster mesh-robot loads.** Distinct collision-mesh hulls are now computed
  in parallel up front (scoped std threads, no new dependency) and the link
  walk takes cache hits: SO-101 with real meshes loads in 5.0 ms release
  (was 16.3) and 0.42 s debug (was 2.70) — bit-identical hulls, pinned by a
  determinism test comparing the primed path against the serial one.
- **CI: lerobot pairing watch + Linux runtime job.** `pairing-watch.yml`
  runs monthly against the *newest* lerobot from PyPI (unpinned, py3.12) and
  fails on any skipped gate — a moved converter or changed import shape turns
  red instead of silently skipping. The new `linux` job in `ci.yml` runs the
  Rust workspace plus the built wheel + core oracle on `ubuntu-latest` on
  every push.

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
