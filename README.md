# Caliper

**A modern, open robotics engine — kinematics · IK · singularity · dynamics · motion · planning · collision · real-robot control · a Simulink-style dataflow studio. One Rust engine, three faces.**

Caliper is a single deterministic Rust engine for serial-arm robotics, exposed
through three faces that share the exact same code:

- **CLI** — `caliper fk | ik | analyze | move | plan | sim | record | graph …`
- **Python** — `import caliper` (built with [maturin](https://www.maturin.rs/)), scriptable like MATLAB/NumPy
- **Studio** — *Caliper Studio*, a Tauri + React desktop app with a 3D scene and a Simulink-style dataflow graph editor

The engine math is cross-validated against [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
and NumPy (residuals down to ~1e-9…1e-15), and the headless stack (engine + CLI +
Python) has been through an independent first-principles re-derivation and a large
multi-agent correctness/safety audit. See **[docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md)**
for the honest trust map.

> **Status — all 9 phases (0–8) built.** Kinematics/IK/singularity, jerk-limited
> motion (S-curve MOVE_J/L/C), dynamics (RNEA/CRBA/forward-dynamics) + simulation,
> real + simulated robot control, RRT-Connect/RRT* planning + reachability,
> collision, a pure-PyTorch behavior-cloning sidecar (`learn/`), and the Phase-8
> dataflow graph editor. The headless stack is machine-verified; **the Studio GUI
> is build-checked and statically reviewed only — it has not yet been run at
> runtime** (deliberate: build now, human-review later).

---

## Features

| Phase | Capability |
|------|------------|
| **0–1 Kinematics** | URDF → frozen kinematic model; forward kinematics; geometric Jacobians (world/body); SE(3)/SO(3) screw math (`exp6`/`log6`, adjoints, spatial inertia) |
| **2 IK & singularity** | Damped-least-squares / Levenberg–Marquardt CLIK with manipulability-gated damping, step clamping, joint limits, multi-restart; **analytic 6R IK**; singular-value / manipulability / condition-number analysis |
| **3 Motion** | Jerk-limited 7-segment S-curve trajectories; time-synchronized **MOVE_J**, Cartesian **MOVE_L / MOVE_C**; waypoint retiming; O(1) closed-form `sample(t)` |
| **4 Dynamics** | Inverse dynamics (**RNEA**), joint-space mass matrix (**CRBA**), forward dynamics, semi-implicit-Euler `Simulator` with gravity |
| **5 Real robots** | Real `RobotBackend` contract; computed-torque `ControlLoop`; `SafetyMonitor`; teleop (leader–follower); LeRobotDataset v2.1 record/replay; feature-gated CAN / Dynamixel hardware skeletons |
| **6 Planning** | **RRT-Connect** + **RRT\*** (deterministic, seeded), shortcut smoothing, collision-aware reachability, trajectory retiming |
| **7 Learning** | `learn/caliper_learn` — pure-PyTorch behavior-cloning sidecar (BC-MLP / ACT-lite / optional DDPM), goal-conditioned, zero `lerobot` runtime dependency |
| **8 Studio graph** | `caliper-graph` serde dataflow IR + deterministic executor (11 node kinds) + a Simulink-style node editor face (run on all three faces) |
| **Collision** | Self-contained, pure-nalgebra checker: OBB↔OBB via separating-axis theorem, sphere/box/half-space closed forms, mesh-as-convex-hull via GJK; honest `uncovered_frames` reporting |

**Wave-1/2 hardening additions:** external Pinocchio/NumPy cross-validation, an
analytic 6R IK path, mesh (convex-hull) collision, and an RRT\* planner.

---

## Verification

Caliper is built "verify as you go". The trust story, in short:

- **External cross-validation** — FK, geometric Jacobians, RNEA, CRBA, forward
  dynamics, and singularity metrics are checked against **Pinocchio** and **NumPy
  SVD** in a Python oracle (`oracle/`), with residuals ~1e-9…1e-15. The oracle
  runs *through the PyO3 bindings*, so it validates the shipped Python face too.
  Motion profiles are sanity-checked against **Ruckig**-class jerk-limited
  expectations; learning/eval data against the LeRobot schema; control against
  SciPy/NumPy where a closed form exists.
- **Property tests** — proptest-style invariants on the math (round-trips,
  monotonicity, endpoint-exactness).
- **Re-derivation + audit** — an independent first-principles re-derivation of all
  11 algorithm clusters plus a multi-agent correctness/safety audit; every
  confirmed finding fixed or documented.

What is **not** runtime-verified: the **Studio GUI** (Tauri backend + React/r3f
frontend) compiles, type-checks, builds, and was statically reviewed, but has
never been launched. Full details, residuals, and the honest gap list are in
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
robot = caliper.Robot("robot.urdf")
q = robot.ik(target, seed)        # target: 4x4 column-major pose
pose = robot.fk(q)
```

> ⚠️ The Python Cartesian-pose entry points are not yet unified: `Robot.ik` /
> `Robot.move_l` take a **4×4 column-major** pose and a frame **name**, while
> `Planner.plan_to_pose` takes a **flat 12-element row-major** pose and a frame
> **index**. This is a documented wart; unifying it is future work.

### Studio (desktop app)

```sh
cd apps/studio
npm install
env -u CONDA_PREFIX npm run tauri dev                 # or, from the repo root: just app
```

This launches *Caliper Studio* (3D scene + dataflow Graph tab). **Heads-up:** the
GUI is build-checked only and has not yet been exercised at runtime — treat the
first launch as the real test.

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

## License

The plan is a split license matching the kind of artifact:

- **Software** — Apache-2.0 (the Rust engine, faces, and tooling).
- **Hardware** — CERN-OHL-W (any open-hardware designs).
- **Documentation** — CC-BY (docs and written material).

These are repo-level `LICENSE` files per artifact type, not per-crate license
fields. (License files are being finalized; until then, crate metadata carries a
permissive `MIT OR Apache-2.0` placeholder.)
