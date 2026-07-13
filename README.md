# Caliper

**Download one file, and a robot is on your screen in under a minute — jog it,
plan for it, simulate it, record a dataset, train a policy, and get a
plain-English verdict on why it did or didn't work. No ROS workspace, no GPU,
no cloud.** ¹

Caliper is a single deterministic Rust engine for serial-arm robotics, exposed
through three faces that share the exact same code:

- **CLI** — `caliper fetch | fk | ik | analyze | move | plan | sim | record | doctor | report | graph …`
- **Python** — `import caliper` (built with [maturin](https://www.maturin.rs/)), scriptable like MATLAB/NumPy
- **Studio** — *Caliper Studio*, a Tauri + React desktop app with a 3D scene and a Simulink-style dataflow graph editor

**Start here → [Zero to moving in 10 minutes](docs/book/src/quickstart.md)** —
one guided path through the whole loop on a real SO-101 description. Full
documentation: **[msannikov03.github.io/caliper](https://msannikov03.github.io/caliper/)**
(the mdBook published from `docs/book/` on every push to main).

## Two fronts, one bet

Robotics software today makes you choose between a datacenter and a junk
drawer. Caliper's bet is that most arm work needs neither:

| | GPU-sim platforms (Isaac Sim/Lab) | The duct-tape stack (ROS + MoveIt + URDF scripts + notebooks) | **Caliper** |
|---|---|---|---|
| Install | tens of GB + a driver matrix | a workspace, a distro pin, an env per tool | one signed `.dmg`, or `cargo build` / `maturin develop` |
| Hardware floor | workstation-class RTX GPU | a Linux box you're afraid to update | a laptop — CPU only ([measured](docs/book/src/reference/lightweight.md)) |
| Robot on screen | after the download and the launcher | after the launch files agree | under a minute ¹ |
| The loop (load → plan → sim → record → train → judge) | sim + learning; bring your own everything else | five tools, five configs, five data formats | one engine, one artifact, three faces |
| When it breaks | a stack trace from someone else's extension | silence, or a crash three tools downstream | a **doctor's report** naming the defect and the fix |
| Trust story | closed components | "it ran on my machine" | [cross-validated to ~1e-9…1e-15](docs/VERIFICATION_REPORT.md), deterministic by construction |

Lighter than the giants, more legible than the duct tape — and that second
front is the one nobody ships: Caliper assumes your robot description is
broken, your dataset is flawed, and your trained policy will do nothing on
deploy, and instruments all three.

## It tells you *why*

The prevailing workflow in robot learning is: export a URDF from CAD, record
some demos, and — as one practitioner put it — *everybody just starts training
and hopes for the best*. Caliper replaces hope with reports:

- **Asset doctor** (`caliper doctor`, `A001`–`A014`) — inertia a converter
  dropped, meshes that don't resolve, limits that can't move: diagnosed in one
  pass, mechanically repaired into a *copy* on request. Runs automatically on
  every Studio load.
- **Dataset doctor** (`caliper data doctor`, `D001`–`D015`) — variance
  collapse, stale normalization stats, contradictory demos, dead cameras —
  caught *before* the GPU bill, not after.
- **Trajectory lint** (`caliper report`, `T001`–`T009`) — limit violations,
  360° detours, jerk spikes, singular corridors, near-misses; `--strict` gates CI.
- **Policy autopsy** (`caliper-learn autopsy`, `E`/`L`/`P` codes) — the trained
  policy's post-mortem: is it a data problem, a model problem, or a deploy-loop
  problem? One report, one verdict paragraph.

Findings are data, not errors; every check has a stable code, a plain-English
message, and a fix hint. See
[Doctors](https://msannikov03.github.io/caliper/capabilities/doctors.html) and
[Verdicts](https://msannikov03.github.io/caliper/capabilities/verdicts.html).

## Measured, not claimed

"Lightweight" ships with numbers or it doesn't ship: install size, cold-start
time, RAM, record overhead — every figure (and every still-`TBD` cell) lives
in **[Lightweight, measured](docs/book/src/reference/lightweight.md)**,
produced by `scripts/measure_lightweight.sh` with machine + git-rev provenance
stamped on. Current headline: the Studio `.dmg` is **10.4 MB** with the MuJoCo
contact engine bundled.

The engine math itself is cross-validated against
[Pinocchio](https://github.com/stack-of-tasks/pinocchio) and NumPy (residuals
down to ~1e-9…1e-15), and the headless stack (engine + CLI + Python) has been
through an independent first-principles re-derivation and a large multi-agent
correctness/safety audit. See
**[docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md)** for the honest
trust map.

¹ **The honest footnotes.** The packaged app is macOS (Apple Silicon) today;
Linux/Windows builds are unproven. The `.dmg` is signed but **not yet
notarized** — first launch needs right-click → Open. "Real-robot control"
means the control loop, safety monitor, teleop and recording stack run against
*simulated* backends; the CAN/Dynamixel hardware codecs are feature-gated
skeletons that have never driven a physical arm. And "under a minute" is a
promise backed by the [metrics page](docs/book/src/reference/lightweight.md) —
where a number is still `TBD`, the page says so instead of rounding hope.

> **Status — all 9 phases (0–8) built, plus the hardening waves on top.**
> Kinematics/IK/singularity, jerk-limited motion (S-curve MOVE_J/L/C + TOPP
> retiming), dynamics (RNEA/CRBA/forward-dynamics) + simulation, real + simulated
> robot control, RRT-Connect/RRT*/PRM planning + CHOMP-style trajectory
> optimization, collision (incl. capsules + EPA contact extraction), joint-offset
> calibration, a pure-PyTorch behavior-cloning sidecar (`learn/`), and a
> Simulink-style dataflow graph editor. **Loads real-world robots** — URDF
> `<visual>` geometry (STL/glTF/COLLADA meshes, `package://` resolution), mimic
> joints, and bounded convex-hull collision meshes; the kinematics are
> cross-validated against Pinocchio on vendored Franka Panda, SO-ARM100
> (SO-100/SO-101), and Kinova Gen3 lite URDFs. The headless stack is
> machine-verified end-to-end; the Studio GUI's non-visual logic is covered by a
> headless vitest harness, and the app builds, signs, and launches — the visual
> polish pass is ongoing.

---

## Features

| Phase | Capability |
|------|------------|
| **0–1 Kinematics** | URDF → frozen kinematic model; forward kinematics; geometric Jacobians (world/body); SE(3)/SO(3) screw math (`exp6`/`log6`, adjoints, spatial inertia) |
| **2 IK & singularity** | Damped-least-squares / Levenberg–Marquardt CLIK with manipulability-gated damping, step clamping, joint limits, multi-restart; **analytic 6R IK**; singular-value / manipulability / condition-number analysis |
| **3 Motion** | Jerk-limited 7-segment S-curve trajectories; time-synchronized **MOVE_J**, Cartesian **MOVE_L / MOVE_C** (arc through a via point); **time-optimal (TOPP) retiming**; waypoint retiming; O(1) closed-form `sample(t)` |
| **4 Dynamics** | Inverse dynamics (**RNEA**), joint-space mass matrix (**CRBA**), forward dynamics, centroidal quantities (COM / total mass), semi-implicit-Euler `Simulator` with gravity |
| **5 Real robots** | Real `RobotBackend` contract; computed-torque `ControlLoop` (+ streaming `run_stream`); `SafetyMonitor`; teleop (leader–follower); LeRobotDataset **v3.0 native** record/replay (+ legacy v2.1) — loads directly in `lerobot` >= 0.4, no converter; feature-gated CAN / Dynamixel hardware skeletons |
| **6 Planning** | **RRT-Connect** + **RRT\*** + **PRM** (deterministic, seeded), shortcut smoothing, **CHOMP-style trajectory optimization** (`caliper-trajopt`), collision-aware reachability |
| **7 Learning** | `learn/caliper_learn` — pure-PyTorch behavior-cloning sidecar (BC-MLP / ACT-lite / optional DDPM), goal-conditioned, zero `lerobot` runtime dependency |
| **8 Studio graph** | `caliper-graph` serde dataflow IR + deterministic executor (11 node kinds) + a Simulink-style node editor face (run on all three faces) |
| **Real-world URDFs** | `<visual>` geometry (primitives + STL/glTF/COLLADA meshes, inline/named materials), `package://` resolution (ancestor search + `CALIPER_PACKAGE_PATH`), **mimic joints** (reduced-space FK/Jacobian via the chain rule), bounded convex-hull collision meshes — validated on Panda / SO-ARM100 / Gen3 lite |
| **Collision** | Self-contained, pure-nalgebra checker: OBB↔OBB via separating-axis theorem, sphere/box/capsule/half-space closed forms, mesh-as-convex-hull via GJK, **EPA penetration contacts**; honest `uncovered_frames` reporting |
| **Calibration** | Joint-offset calibration (damped Gauss–Newton on FK residuals) via `caliper-calib` |
| **Redundancy** | Resolved-rate control + null-space motion for redundant (>6-DOF) arms |

**Studio daily-driver features:** ⌘K command palette + keyboard shortcuts,
File → Open any URDF with recents, session resume (window + robot + pose + mode),
rotating file logs + panic capture, graph editor with delete/duplicate/fit and
shareable `.caliper-graph.json` file export/import.

---

## Verification

Caliper is built "verify as you go". The trust story, in short:

- **External cross-validation** — FK, geometric Jacobians, RNEA, CRBA, forward
  dynamics, centroidal quantities, and singularity metrics are checked against
  **Pinocchio** and **NumPy SVD** in a Python oracle (`oracle/`, 100+ tests, zero
  skips), with residuals ~1e-9…1e-15 — including on **real vendored robot URDFs**
  (Franka Panda, SO-100/SO-101, Kinova Gen3 lite), not just hand-authored
  fixtures. The oracle runs *through the PyO3 bindings*, so it validates the
  shipped Python face too. Motion profiles are cross-checked against **Ruckig**;
  `log6` against **SciPy** `logm`; learning/eval data against the LeRobot schema.
- **Property tests** — proptest-style invariants on the math (round-trips,
  monotonicity, endpoint-exactness).
- **Re-derivation + audit** — an independent first-principles re-derivation of all
  11 algorithm clusters plus a multi-agent correctness/safety audit; every
  confirmed finding fixed or documented.

The **Studio GUI**'s non-visual half (store logic, graph serialization ↔ Rust
schema contract, coordinate conventions, palette/command model, session
persistence) is pinned by a **headless vitest harness (130+ tests, in CI)**; the
app builds, code-signs, and launches. What is *not* machine-verified is the
rendered pixels — the visual/UX review is a human pass. Full details, residuals,
and the honest gap list are in
**[docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md)**.

---

## Quickstart

Requires a recent stable Rust (edition 2024; MSRV 1.89). The optional
[`just`](https://github.com/casey/just) recipes below mirror the raw commands.

### CLI

```sh
cargo run -p caliper-cli -- info
cargo run -p caliper-cli -- fk    robot.urdf --joints 0.1,0.2,0.0,0.0,0.0,0.0
cargo run -p caliper-cli -- ik    robot.urdf --target 1,0,0,0,1,0,0,0,1,0.3,0.0,0.2
cargo run -p caliper-cli -- move  robot.urdf --target 1,0,0,0,1,0,0,0,1,0.3,0.0,0.2
cargo run -p caliper-cli -- plan  robot.urdf --goal 0.5,0.2,-0.3,0,0,0 --ground 0.0
cargo run -p caliper-cli -- graph robot.urdf my.caliper-graph.json
```

### Python (via maturin)

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop -m crates/caliper-py/Cargo.toml      # or: just py
```

```python
import caliper
robot = caliper.Robot.from_urdf("robot.urdf")
q = robot.ik(target, seed)          # target: 4x4 column-major pose
pose = robot.fk(q)                  # world pose of the tip frame
J = robot.jacobian(q)               # 6xN geometric Jacobian, [v; ω]
traj = robot.move_l(q, target)      # jerk-limited Cartesian line
branches = robot.analytic_ik(target)  # closed-form 6R IK (when canonical)
```

The bindings ship **PEP 561 type stubs** (`.pyi`), so `import caliper` is fully
typed in your editor.

> The Python Cartesian-pose entry points share ONE convention: every
> pose-accepting method (`Robot.ik` / `analytic_ik` / `move_l` / `move_c`,
> `Planner.plan_to_pose`, `ReachChecker.status` / `reachable`,
> `calibrate_joint_offsets`) takes a **4×4 column-major** pose — nested
> (`pose[col][row]`) or its flat 16-element equivalent — and every frame
> argument takes a frame **name** (a raw index still works where one was
> accepted before). `Planner.plan_to_pose` additionally grandfathers the
> legacy flat 12-element row-major form for back-compat. (`Robot.fk()`
> returns **row-major** — transpose to feed it back into `ik()`.)

### Studio (desktop app)

**Install (macOS, Apple Silicon):** grab the `.dmg` from the
[Releases](https://github.com/msannikov03/caliper/releases) page. The app is
signed with an Apple **Development** certificate (not yet notarized for
distribution), so on first open Gatekeeper will object — right-click the app →
**Open** → Open, or allow it under *System Settings → Privacy & Security*.

**Or run from source:**

```sh
cd apps/studio
npm install
npm run tauri dev                                     # or, from the repo root: just app
```

This launches *Caliper Studio*: a 3D scene rendering the robot's real `<visual>`
geometry, jog/motion/simulate modes, drag-IK with a singularity HUD, and the
dataflow Graph tab. Press **⌘K** for the command palette.

---

## Crate map

The workspace is a set of small, focused crates. `caliper` is the umbrella that
re-exports the engine; the three faces build on it.

| Crate | Role |
|------|------|
| `caliper-spatial` | SE(3)/SO(3) screw math — twists, `exp6`/`log6`, adjoints, spatial inertia (`[v; ω]` order, Pinocchio-compatible) |
| `caliper-model` | URDF parsing → frozen struct-of-arrays kinematic `Model` |
| `caliper-kinematics` | Forward kinematics, Jacobians, singularity analysis |
| `caliper-ik` | Inverse kinematics (DLS/LM CLIK + analytic 6R) |
| `caliper-dynamics` | RNEA / CRBA / forward dynamics + `Simulator` |
| `caliper-motion` | Jerk-limited S-curve trajectories (MOVE_J/L/C) + retiming |
| `caliper-planning` | RRT-Connect / RRT\* planner, smoothing, reachability |
| `caliper-collision` | Pure-nalgebra collision checker (OBB-SAT, GJK, half-space) |
| `caliper-hal` | `RobotBackend` trait, control loop, safety, teleop, dataset, HW codecs |
| `caliper-graph` | Phase-8 dataflow IR + deterministic graph executor |
| `caliper` | Umbrella facade re-exporting the engine modules |
| `caliper-cli` | Command-line face |
| `caliper-py` | Python bindings (`import caliper`, PyO3) |
| `apps/studio` | Tauri desktop app — *Caliper Studio* |
| `learn/` | Phase-7 pure-PyTorch behavior-cloning sidecar (`caliper_learn`) |

`apps/studio` is excluded from the default workspace build (it needs a built
frontend); use `just app` / `npm run tauri dev`.

---

## Development

```sh
just ci        # fmt-check + clippy + test + lean-core check
just test      # cargo test --workspace --exclude studio
just oracle    # Pinocchio/NumPy cross-validation (needs the repo .venv)
just learn     # pure-PyTorch BC sidecar tests
```

The oracle and learning tests need a Python venv with `maturin`, `pinocchio`,
`numpy`, `pyarrow` (and `torch` for `learn`) installed — see the recipe comments
in the [`justfile`](justfile).

---

## Docs

A full mdBook docs site lives in [`docs/book`](docs/book) (architecture,
per-capability guides, the verification story) — `mdbook build docs/book`, or
read the markdown directly.

---

## License

Split licensing by artifact type (see the repo-level license files):

- **Software** — [Apache-2.0](LICENSE-APACHE) (the Rust engine, faces, and tooling).
- **Hardware** — [CERN-OHL-W](LICENSE-CERN-OHL-W) (any open-hardware designs).
- **Documentation** — [CC-BY](LICENSE-CC-BY) (docs and written material).
