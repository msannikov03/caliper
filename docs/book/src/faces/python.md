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

## Known wart: pose conventions are not yet unified

The Cartesian-pose entry points are **not** uniform, and this is documented
honestly rather than hidden:

- `Robot.ik` / `Robot.move_l` take a **4×4 column-major** pose and a frame
  **name**.
- `Planner.plan_to_pose` takes a **flat 12-element row-major** pose and a frame
  **index**.

Unifying these signatures is future work; until then, mind which entry point you
are calling.
