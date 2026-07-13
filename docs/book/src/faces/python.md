# Python (maturin)

The `caliper-py` face builds the engine into a native Python extension with
[maturin](https://www.maturin.rs/) / PyO3, so you can `import caliper` and script
the engine like NumPy/MATLAB. This is also the surface the **oracle** runs
through — validating FK, Jacobians, RNEA, CRBA, forward dynamics, and singularity
metrics against Pinocchio/NumPy exercises the *shipped Python bindings*, not a
private test path.

## Build

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop -m crates/caliper-py/Cargo.toml
```

(In this repo the convention is `env -u CONDA_PREFIX .venv/bin/maturin develop
-m crates/caliper-py/Cargo.toml`, building into the repo `.venv`.)

## Use

```python
import caliper

robot = caliper.Robot.from_urdf("robot.urdf")
res   = robot.ik(target, seed)   # target: a 4x4 column-major pose; seed: joints
pose  = robot.fk(res["q"])       # ik returns {success, q, residual, iters, …}
```

Beyond `Robot`, the bindings expose `Planner`, `ControlLoop` (with
`step_with_target` and `last_warn`), `Recorder`, `DatasetReader`, and a
`run_graph` entry point for the dataflow graph — this is the surface the
[learning sidecar](../capabilities/learning.md) builds on. The full surface
is typed in `crates/caliper-py/python/caliper/__init__.pyi`; the [capability
matrix](../reference/capability-matrix.md) maps every function to its engine
capability.

## Doctors & lint

The three diagnostic engines (see [Doctors & trajectory
lint](../capabilities/doctors.md)) are plain functions returning plain data —
findings never raise; only an uninspectable input does:

```python
import caliper

# Asset doctor: A001–A014 over a URDF/xacro. Findings are dicts with
# {code, severity ("error"|"warn"|"info"), message, fix_hint, auto_fixable}.
rep = caliper.doctor("robot.urdf")
assert rep["clean"] or rep["errors"] == 0

# repair=True writes a repaired COPY (default <input>.repaired.urdf; the
# input is never modified) and reports {out, applied, skipped, mesh_copies}.
rep = caliper.doctor("robot.urdf", repair=True, density=2700.0)
fixed = rep["repair"]["out"]
assert caliper.doctor(fixed)["clean"]          # findings describe the ORIGINAL

# Dataset doctor: D001–D015 over a LeRobotDataset v3.0 root. Also returns the
# recomputed per-feature stats {dim, mean, std, min, max, bin_occupancy}.
dr = caliper.data_doctor("~/datasets/pick_place")
for f in dr["findings"]:
    print(f["severity"], f["code"], f["message"])

# Trajectory lint: T001–T007 over sampled rows (exactly what
# Trajectory.sample_uniform returns); [] means the trajectory lints clean.
robot = caliper.Robot.from_urdf("robot.urdf")
goal = [0.5] * robot.ndof
traj = robot.move_j([0.0] * robot.ndof, goal)
times, q, qd, qdd = traj.sample_uniform(0.01)
findings = caliper.lint_path(robot, times, q, qd, qdd)
```

(The collision half of the lint, `T008`/`T009`, is CLI-only — `caliper
report`.)

## Pose convention: unified

Every pose-accepting entry point (`Robot.ik` / `analytic_ik` / `move_l` /
`move_c`, `Planner.plan_to_pose`, `ReachChecker.status` / `reachable`,
`calibrate_joint_offsets`) takes the **same** input: a **4×4 column-major**
nested list (or an equivalent flat 16-element column-major list), and frame
arguments accept a **name** everywhere (an integer index is still accepted
where it historically was, now bounds-checked).

One legacy form is grandfathered for back-compat: `Planner.plan_to_pose` also
accepts its original flat 12-element row-major pose (9 rotation entries then
`tx, ty, tz`). New code should use the 4×4 form.
