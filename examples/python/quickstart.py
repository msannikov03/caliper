#!/usr/bin/env python3
"""Caliper Python quickstart — load a URDF, do FK / IK / gravity torque.

This mirrors the Rust `load_and_fk` + `ik_demo` examples through the `caliper`
Python binding (the scripting / analysis face of the engine).

Build the binding first (from the repo root), then run this script:

    # one-time: create a venv and install maturin
    python3 -m venv .venv && . .venv/bin/activate
    pip install maturin

    # build + install the `caliper` extension module into the active venv
    maturin develop -m crates/caliper-py/Cargo.toml

    # run
    python examples/python/quickstart.py

The module name is `caliper` (see crates/caliper-py/pyproject.toml).
"""

from pathlib import Path

import caliper

# Oracle test fixture, located relative to this script (repo_root/oracle/...).
REPO_ROOT = Path(__file__).resolve().parents[2]
URDF = REPO_ROOT / "oracle" / "fixtures" / "robots" / "showcase6.urdf"


def transpose4(m):
    """Row-major 4x4 (as fk() returns) -> column-major (as ik() expects)."""
    return [[m[r][c] for r in range(4)] for c in range(4)]


def main() -> None:
    print("caliper", caliper.__version__)

    robot = caliper.Robot.from_urdf(str(URDF))
    print(f"loaded `{robot.name}` — {robot.ndof} DOF")
    print("joints:", robot.joint_names)
    print("tool frame:", robot.tip_frame())

    # Forward kinematics (4x4 row-major homogeneous transform of the tip).
    q_true = [0.4, -0.6, 0.9, 0.2, -0.5, 0.3]
    pose = robot.fk(q_true)
    tip = [pose[0][3], pose[1][3], pose[2][3]]
    print("tip position:", [round(v, 4) for v in tip])

    # Inverse kinematics: recover a configuration reaching that pose.
    # fk() is row-major; ik() wants a column-major 4x4 -> transpose.
    res = robot.ik(transpose4(pose), seed=[0.0] * robot.ndof)
    print(f"ik success={res['success']} residual={res['residual']:.2e}")
    if res["success"]:
        reached = robot.fk(res["q"])
        err = sum((reached[i][3] - tip[i]) ** 2 for i in range(3)) ** 0.5
        print(f"ik position error: {err:.2e} m")

    # Inverse dynamics: static gravity-hold torque at q_true.
    # `has_inertia` is a property, not a method.
    if robot.has_inertia:
        tau = robot.gravity_torque(q_true)
        print("gravity torque [N·m]:", [round(t, 4) for t in tau])


if __name__ == "__main__":
    main()
