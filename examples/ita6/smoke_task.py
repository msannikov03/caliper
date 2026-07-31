#!/usr/bin/env python3
"""Smoke test: ITA-6 through the FULL caliper stack.

Loads `ita6.caliper-task.json`, builds the MuJoCo vector env from it and steps
it -- the proof that the designed arm is not just a URDF that parses, but a
robot the engine can actually simulate against a scored task.

    PATH=/usr/sbin:$PATH .venv/bin/python examples/ita6/smoke_task.py
"""

from pathlib import Path

import numpy as np

from caliper_learn import VecSimEnv, load_task

HERE = Path(__file__).resolve().parent
STEPS = 5


def main() -> None:
    task = load_task(str(HERE / "ita6.caliper-task.json"))
    print(f"task           {task.name}")
    print(f"robot          {Path(task.robot).name}")
    print(f"props / zones  {[p['name'] for p in task.props]} / "
          f"{[z['name'] for z in task.zones]}")
    print(f"gripper joint  {task.gripper}")
    print(f"success        {task.success}")

    # `init_jitter=0` starts exactly at the task's q0 -- the default 0.2
    # randomises the start by 20 % of each range, which is what you want for
    # data collection and not what you want for a hold check. Default gains
    # are kept: the env's law is gravity-compensated, so ITA-6 holds q0 to
    # 0.00 mrad over 2 s at kp=100 even needing ~6 N.m at the shoulder.
    env = VecSimEnv.from_task(task, num_envs=2, init_jitter=0.0)
    obs = env.reset(seed=0)
    print(f"\nenv            {env.num_envs} envs, {env.ndof} dof, "
          f"obs keys {sorted(obs)}")

    # obs["state"] is [q | qd] per env; actions are qpos targets for the robot.
    q = np.array(obs["state"][:, : env.ndof], dtype=np.float64)
    for i in range(STEPS):
        obs, reward, terminated, truncated, info = env.step(q)   # hold q0
        q = np.array(obs["state"][:, : env.ndof], dtype=np.float64)
        assert np.all(np.isfinite(q)), f"non-finite q at step {i}"
    print(f"stepped        {STEPS} steps, q finite")
    print(f"q[0] after     {np.round(q[0], 4).tolist()}")
    print(f"success flag   {info.get('success')}")

    # And it HOLDS. A gravity-compensated arm that drifts is usually a geometry
    # bug in disguise: an early ITA-6 wrist self-collided with the forearm, and
    # this check is what made it visible.
    tgt = np.tile(np.array(task.q0, dtype=np.float64), (env.num_envs, 1))
    obs = env.reset(seed=0)
    for _ in range(2 * task.fps):
        obs, *_ = env.step(tgt)
    droop = float(np.abs(np.array(obs["state"][:, : env.ndof]) - tgt).max())
    print(f"hold 2 s       max |q - q0| = {droop * 1000:.2f} mrad")
    assert droop < 5e-3, f"arm does not hold its start pose: {droop:.4f} rad"

    print("\nOK")


if __name__ == "__main__":
    main()
