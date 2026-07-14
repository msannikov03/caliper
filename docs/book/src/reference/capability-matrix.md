# Capability matrix

Everything the system can do, capability by capability, against **where you
can do it**. Compiled by reading the actual surfaces — the clap verb enum in
`crates/caliper-cli/src/main.rs`, the typed Python surface in
`crates/caliper-py/python/caliper/__init__.pyi`, the sidecar exports in
`learn/caliper_learn/__init__.py` (+ its `caliper-learn` console script), and
the Studio modes/store in `apps/studio/src` — not from memory. A ✗ is an
honest gap: the engine can do it, that face does not expose it (yet).

Faces: [CLI](../faces/cli.md) · [Python](../faces/python.md) ·
[Studio](../faces/studio.md). Engine column links to the capability page.

## Model & assets

| Capability | Engine | CLI | Python | Studio |
|---|---|---|---|---|
| Load URDF / xacro, summarize structure | `caliper-model` ([architecture](../architecture.md)) | `load` | `Robot.from_urdf`, `.name/.ndof/.joint_names/.joint_limits/.frame_names/.tip_frame/.has_inertia` | Open URDF… / samples / recents, ⌘O |
| **Asset doctor** — diagnose `A001`–`A014` | [`caliper-doctor`](../capabilities/doctors.md) | `doctor` | `doctor(path)` | automatic on every load (error-banner findings) |
| **Asset repair** — repaired copy, never in-place | [`caliper-doctor`](../capabilities/doctors.md) | `doctor --repair [--density] [--out]` | `doctor(path, repair=True, density=…)` | **Repair & reload** button |
| MJCF (MuJoCo XML) export (+ hull-mesh assets) | [`caliper-sim-mujoco::mjcf`](../capabilities/contact-sim.md) | `mjcf` (`--hull-meshes`) | `model_to_mjcf` | ✗ |
| **Robot zoo** — fetch a vendored real URDF | `caliper-cli::zoo` | `fetch <name>` / `fetch --list` | ✗ | ✗ |

## Kinematics & analysis

| Capability | Engine | CLI | Python | Studio |
|---|---|---|---|---|
| Forward kinematics (every frame) | [`caliper-kinematics`](../capabilities/kinematics.md) | `fk` | `Robot.fk` | live in every mode (Jog sliders drive it) |
| Geometric Jacobian (world/body) | [`caliper-kinematics`](../capabilities/kinematics.md) | ✗ | `Robot.jacobian` | ✗ (internal to the HUD analysis) |
| Iterative IK (DLS/LM, restarts) | [`caliper-ik`](../capabilities/kinematics.md) | `ik` | `Robot.ik` | Jog tip gizmo (singularity-governed) |
| Analytic 6R IK (branch set) | [`caliper-ik::analytic`](../capabilities/kinematics.md) | `ik --analytic` | `Robot.analytic_ik` | ✗ |
| Singularity / manipulability analysis | [`caliper-kinematics`](../capabilities/kinematics.md) | `analyze` | `Robot.analyze` / `manipulability` / `ellipsoid` | singularity HUD + manipulability ellipsoid |
| Redundancy: nullspace step, resolved-rate | [`caliper-kinematics`](../capabilities/kinematics.md) | ✗ | `Robot.nullspace_step` / `Robot.resolved_rate` | ✗ |
| SE(3) log/exp maps | `caliper-spatial` | ✗ | `log6` / `exp6` | ✗ |

## Motion

| Capability | Engine | CLI | Python | Studio |
|---|---|---|---|---|
| MOVE_J (jerk-limited S-curve) | [`caliper-motion`](../capabilities/motion.md) | `move --goal` | `Robot.move_j` | Motion mode / palette "Plan move to home" / poses |
| MOVE_L (Cartesian line) | [`caliper-motion`](../capabilities/motion.md) | `move --target` | `Robot.move_l` | Motion mode (gizmo target) |
| MOVE_C (circular arc via a point) | [`caliper-motion`](../capabilities/motion.md) | `move --target --via` | `Robot.move_c` | ✗ |
| Waypoint retiming | [`caliper-motion`](../capabilities/motion.md) | ✗ | `Planner.plan_trajectory` (plan → retimed `Trajectory`) | ✗ |
| Time-optimal (TOPP) retiming | [`caliper-motion`](../capabilities/motion.md) | `move --time-optimal` | `Robot.retime_time_optimal` | ✗ |
| Named pose library | `caliper-motion::PoseLibrary` | ✗ | ✗ | Motion mode (save/plan-to/delete poses) |
| **Trajectory lint** `T001`–`T007` | [`caliper-kinematics::lint_path`](../capabilities/doctors.md) | `report` | `lint_path` | ✗ |
| **Collision lint** `T008`/`T009` | [CLI-side over `caliper-collision`](../capabilities/doctors.md) | `report --ground/--obstacle/--clearance [--strict]` | ✗ | ✗ |
| Cycle-time + path-quality report | `caliper-kinematics::path_report` | `report` | ✗ | ✗ |

## Dynamics & simulation

| Capability | Engine | CLI | Python | Studio |
|---|---|---|---|---|
| RNEA / CRBA / forward dynamics / gravity | [`caliper-dynamics`](../capabilities/dynamics.md) | `dyn` | `Robot.rnea` / `crba` / `forward_dynamics` / `gravity_torque` | ✗ ¹ |
| Passive/forced time-stepped simulation | [`caliper-dynamics::Simulator`](../capabilities/dynamics.md) | `sim` | `Simulator` (step/rollout/energy) | Simulate: gravity drop |
| MuJoCo contact simulation (props, ground) | [`caliper-sim-mujoco`](../capabilities/contact-sim.md) | ✗ (use `mjcf` + MuJoCo) | ✗ | Simulate: contact drop / hold / drive-to (mujoco builds) |
| **Contact material presets** (`Rigid`/`Rubber`/`Foam`/`Steel`/`Wood`/`Custom`) | [`caliper-sim-mujoco::mjcf`](./../capabilities/data-factory.md) | via `mjcf` scene | ✗ | ✗ |
| **Contact stability linter** `C001`–`C003` | [`caliper-sim-mujoco::lint`](../capabilities/data-factory.md) | ✗ | ✗ | ✗ (engine-only, mujoco feature) |
| Convex-decomposition seam (identity impl) | [`caliper-sim-mujoco`](../capabilities/data-factory.md) | ✗ | ✗ | ✗ |

## Collision & planning

| Capability | Engine | CLI | Python | Studio |
|---|---|---|---|---|
| Self/world collision query | [`caliper-collision`](../capabilities/collision.md) | `collide` | `CollisionModel.query` | Simulate: Check collision |
| EPA penetration contacts | [`caliper-collision`](../capabilities/collision.md) | `collide --contacts` | `CollisionModel.contacts` | ✗ |
| RRT-Connect (+ shortcut smoothing) | [`caliper-planning`](../capabilities/planning.md) | `plan` | `Planner.plan` / `verify` | Simulate: Plan to home (RRT) |
| RRT\* (asymptotically optimal) | [`caliper-planning`](../capabilities/planning.md) | `plan --optimal` | `Planner.plan_optimal` | ✗ |
| PRM roadmap planning | [`caliper-planning`](../capabilities/planning.md) | `plan --prm` | `Planner.plan_prm` | ✗ |
| Plan to a Cartesian pose | [`caliper-planning`](../capabilities/planning.md) | `plan --target` | `Planner.plan_to_pose` | ✗ |
| CHOMP-style trajectory optimization | [`caliper-trajopt`](../capabilities/planning.md) | ✗ | ✗ | ✗ (engine-only) |
| Collision-aware reachability | [`caliper-planning::reach`](../capabilities/planning.md) | `reach` | `ReachChecker.status` / `reachable` | ✗ ¹ |

## Control, data & learning

| Capability | Engine | CLI | Python | Studio |
|---|---|---|---|---|
| Computed-torque control loop | [`caliper-hal`](../capabilities/control-safety.md) | `run` | `ControlLoop` (`run_to` / `rollout_to` / `step_with_target` / `run_stream`) | Simulate: Drive to home (control) |
| Safety monitor (limits gate, e-stop) | [`caliper-hal`](../capabilities/control-safety.md) | inside `run` ² | `SafetyMonitor` | inside the control rollout ² |
| Leader–follower teleop | [`caliper-hal`](../capabilities/control-safety.md) | `teleop` | `LeaderFollower` | ✗ |
| Dataset record (LeRobotDataset) | [`caliper-dataset`](../capabilities/control-safety.md) | `record` (v3.0 / `--format v21`) | `RecorderV3` / `Recorder` | ✗ |
| Dataset replay through a sim backend | `caliper-dataset` + `caliper-hal` | `replay` | ✗ (readers only) | ✗ |
| Dataset read / browse / plot | `caliper-dataset` | ✗ | `DatasetReaderV3` / `DatasetReader` (incl. images) | Data mode (table, channel plots, thumbnails) |
| Dataset edit: delete / split / merge / tags | `caliper-dataset::edit` | ✗ | `dataset_delete_episodes` / `dataset_split_episode` / `dataset_merge_episodes` / `dataset_read_tags` / `dataset_write_tags` | Data mode edit bar + tag chips |
| **Dataset doctor** `D001`–`D015` | [`caliper-dataset::analyze`](../capabilities/doctors.md) | `data doctor` | `data_doctor` | Data mode: Doctor button (episode-jump findings) |
| Joint-offset calibration (Gauss-Newton) | [`caliper-calib`](../capabilities/calibration.md) | `calibrate` (incl. `--self-test`) | `calibrate_joint_offsets` | ✗ |
| lerobot calibration-file export | Python interop | ✗ | `export_lerobot_calibration` | ✗ |
| robomimic HDF5 export | Python interop | ✗ | `export_robomimic_hdf5` | ✗ |
| BC learning (BC-MLP / ACT-lite / DDPM) | [`learn/caliper_learn` sidecar](../capabilities/learning.md) | ✗ | separate `caliper_learn` package (on top of these bindings) | ✗ |
| **Seeded policy eval** — Wilson-95, `E001`–`E003`, checkpoint `sweep` | [`caliper_learn.eval`](../capabilities/verdicts.md) | ✗ ³ (`caliper-learn eval`) | `evaluate` / `sweep` / `reach_eval_task` | ✗ ³ |
| **Deploy-loop latency profile** — `L001`–`L003`, honest achievable Hz | [`caliper_learn.profile`](../capabilities/verdicts.md) | ✗ ³ (`caliper-learn profile`) | `profile_rollout` | ✗ ³ |
| **Policy deploy debugger** `P001`–`P008` | [`caliper_learn.debugger`](../capabilities/verdicts.md) | ✗ ³ (`caliper-learn debug`) | `analyze_policy` | ✗ ³ |
| **Policy Autopsy** — D+P+E+L under one verdict | [`caliper_learn.autopsy`](../capabilities/verdicts.md) | ✗ ³ (`caliper-learn autopsy`) | `autopsy` | ✗ ³ |
| **Domain randomization** (CI-diffable seeded draws) | [`caliper_learn.randomize`](../capabilities/data-factory.md) | ✗ | `RandomizationSpec` / `sample` / `apply_to_mjcf` / `apply_to_env` / `VecSimEnv(randomization=)` | ✗ |
| **Coverage generator** (doctor→generator loop) | [`caliper_learn.coverage_gen`](../capabilities/data-factory.md) | ✗ ³ (`caliper-learn coverage`) | `generate_coverage` | ✗ |
| Vectorized sim env (gym-vector semantics) | [`caliper_learn.vec_env`](../capabilities/learning.md) | ✗ | `VecSimEnv` / `reach_task` / `rollout_random` | ✗ |
| Sim-camera collector (offscreen → image dataset) | [`caliper_learn.sim_camera`](../capabilities/learning.md) | ✗ | `SimCameraScene` / `collect_camera_dataset` | ✗ |
| **MP4 video features** (dtype `video`, lerobot-exact) | [`caliper_learn.video`](../capabilities/data-factory.md) | ✗ | `encode_episode_video` / `VideoRecorder` / `attach_video_metadata` / `available` | ✗ |

## Dataflow graph

| Capability | Engine | CLI | Python | Studio |
|---|---|---|---|---|
| Run a graph | [`caliper-graph`](../capabilities/studio-graph.md) | `graph run` | `run_graph` | Graph mode: Run (+ live scopes) |
| Validate a graph (types, cycles, topo) | [`caliper-graph`](../capabilities/studio-graph.md) | `graph validate` | `validate_graph` | Graph mode: Validate (inline node/edge errors) |
| Edit a graph visually | — (frontend) | ✗ | ✗ | Graph mode editor (⌘D duplicate, delete, fit, app-data save/load, file import/export) |

## Misc

| Capability | Engine | CLI | Python | Studio |
|---|---|---|---|---|
| Engine version / build info | `caliper::VERSION` | `info` / `--version` | `version()` / `__version__` | toolbar readout |
| First-run guided tour | — (frontend) | ✗ | ✗ | 6-step overlay + palette "Show tour" |
| Lightweight benchmark harness | — (scripts) | `scripts/measure_lightweight.sh` | ✗ | ✗ |

---

¹ The Studio backend registers `dynamics_at` / `reach_check` commands, but no
UI control invokes them yet — counted as ✗ until a panel drives them.

² The safety monitor runs inside the control loop on these faces; only Python
exposes it as a standalone object.

³ Policy inference is Python-side (torch + the safetensors-only `hub` loader),
so the Rust CLI and Studio cannot host these. The sidecar ships its own console
face instead: **`caliper-learn debug|autopsy|eval|profile`** (each takes
`--json`; exit code 1 on any error-severity finding) — see
[Verdicts](../capabilities/verdicts.md).
