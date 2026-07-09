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

robot = caliper.Robot("robot.urdf")
q     = robot.ik(target, seed)   # target: a 4x4 column-major pose; seed: joints
pose  = robot.fk(q)
```

Beyond `Robot`, the bindings expose `Planner`, `ControlLoop` (with
`step_with_target` and `last_warn`), `Recorder`, `DatasetReader`, and a
`run_graph` entry point for the dataflow graph — this is the surface the
[learning sidecar](../capabilities/learning.md) builds on.

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
