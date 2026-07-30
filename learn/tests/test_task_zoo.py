"""The TASK ZOO — every `tasks/*.caliper-task.json` at the repo root, run.

`crates/caliper-sim-mujoco/tests/task_zoo.rs` sweeps the same files through the
Rust loader; it can prove a task file is well-formed, reachable and not already
solved, but not that its scene COMPILES. That is this file's job: each zoo task
is built into a real `VecSimEnv` and stepped, so "works out of the box" means
the MJCF assembled, the props became free bodies, the start pose was inside the
joint limits, and the success predicate answered a live state instead of
raising on it.

Fast on purpose — one env, five steps per task. The zoo is a smoke lane, not a
benchmark.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
caliper = pytest.importorskip("caliper")

from caliper_learn.success import SuccessState, clone  # noqa: E402
from caliper_learn.task import TASK_VERSION, load_task  # noqa: E402
from caliper_learn.vec_env import VecSimEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ZOO = ROOT / "tasks"

# The gripper opening every witness pose below is evaluated at — the zoo's own
# `q0` third entry, i.e. the jaw where the tasks start it.
GRIPPER_Q = 0.02

# One WITNESS per task: (prop to carry, the joint pose that grasps it, a joint
# pose whose resulting prop position satisfies the verdict).
#
# These are hand-picked from a sweep of the joint grid and deliberately sit in
# the MIDDLE of the admissible set, not on a zone face, so the assertion below
# fails on a real regression rather than on floating-point noise. They are not
# a solution the robot executes — they are proof one exists.
WITNESS = {
    "01_lift": ("cube", (-0.05, 0.20), (-0.80, -0.20)),
    "02_place": ("cube", (-0.05, 0.20), (0.275, 0.075)),
    "03_hold_high": ("cube", (-0.05, 0.20), (1.30, -0.425)),
    "04_sort": ("tall_block", (0.425, 0.875), (-0.775, 1.375)),
    "05_precise_place": ("cube", (-0.05, 0.20), (0.275, 0.075)),
}


def zoo_files() -> list[Path]:
    files = sorted(ZOO.glob("*.caliper-task.json"))
    # A glob that matched nothing would make every parametrized case vanish
    # silently rather than fail.
    assert len(files) >= 4, f"expected at least 4 zoo tasks, found {len(files)}"
    return files


ZOO_FILES = zoo_files()
IDS = [p.name.removesuffix(".caliper-task.json") for p in ZOO_FILES]


def test_the_zoo_names_every_task_once():
    """A task NAME is how a run is identified in an eval report or a recorded
    episode, so two zoo tasks may not share one."""
    names = [load_task(p).name for p in ZOO_FILES]
    assert len(set(names)) == len(names), f"duplicate task names in the zoo: {names}"


@pytest.mark.parametrize("path", ZOO_FILES, ids=IDS)
def test_zoo_task_loads(path: Path):
    """The python reader agrees with the Rust one about every shipped file."""
    task = load_task(path)
    assert task.name and task.fps
    # In-repo, relative, resolvable on a fresh clone — no fetch step in front.
    assert not Path(task.robot).is_absolute()
    assert task.robot_path.is_file()
    assert task.props, "a zoo task needs something to act on"
    assert task.success is not None, "a zoo task must define a verdict"
    # `horizonS` is what gives the env a step budget rather than an open rollout.
    assert task.horizon_s and task.steps(task.fps) == pytest.approx(
        task.horizon_s * task.fps, abs=1
    )
    assert load_task(path).to_dict()["version"] == TASK_VERSION


@pytest.mark.parametrize("path", ZOO_FILES, ids=IDS)
def test_zoo_task_builds_a_steppable_env_and_is_not_born_solved(path: Path):
    """The real proof: the scene compiles, the props are there, the predicate
    judges live states — and the verdict is FALSE at the start.

    `evaluate()` latches success (one true step marks the whole episode), so a
    task whose predicate already holds at spawn would report 100% for a policy
    that does nothing. The Rust sweep asserts this against the file's declared
    positions; here it is asserted against what MuJoCo actually built.
    """
    task = load_task(path)
    with VecSimEnv.from_task(task, num_envs=1) as env:
        assert env.model.nq == env.ndof + 7 * len(task.props)
        obs = env.reset(seed=0)
        assert obs["state"].shape == (1, 2 * env.ndof)

        # Every prop the file declares is addressable, where the file put it.
        state = env.success_state(0)
        for prop in task.props:
            assert state.pos(prop["name"]) == pytest.approx(prop["pos"], abs=1e-9)

        # Not born solved. Judging the reset state ANSWERS (it does not raise:
        # a live state always carries velocities, which `settled_speed` needs).
        # A clone, because a predicate captures a baseline when it is reset and
        # `env.success_predicate is task.success`.
        spawn_verdict = clone(task.success)
        spawn_verdict.reset(state)
        assert not spawn_verdict(state), (
            f"{path.name} is BORN SOLVED — {task.success.describe()} already holds "
            "at the start pose, so a do-nothing policy scores 100%"
        )

        # ... and it keeps answering once physics is running.
        hold = obs["state"][:, : env.ndof].astype(np.float64)
        for _ in range(5):
            _o, _r, _te, _tr, info = env.step(hold)
            assert info["success"].shape == (1,)
        assert not info["success"].any(), (
            f"{path.name} succeeds while the arm holds its start pose"
        )


@pytest.mark.parametrize("path", ZOO_FILES, ids=IDS)
def test_zoo_task_is_actually_solvable(path: Path):
    """A verdict no reachable pose can satisfy is a task nobody can pass.

    Nothing in the schema catches that: `all_of [lifted 0.05 above start,
    settled in a bin BELOW the start]` loads cleanly and can never fire. So the
    zoo pins a witness per task — grasp the prop at a pose whose jaw really
    touches its top face, carry it with the weld transform that grasp implies,
    and the predicate must return True at the far end. Real forward kinematics,
    so a zone edited out of the arm's reach fails here.
    """
    task = load_task(path)
    prop, q_grasp, q_hold = WITNESS[path.name.removesuffix(".caliper-task.json")]

    robot = caliper.Robot.from_urdf(str(task.robot_path))
    spawn = {p["name"]: tuple(float(v) for v in p["pos"]) for p in task.props}
    carried = next(p for p in task.props if p["name"] == prop)

    # The grasp really is a grasp: the jaw's bottom face meets the prop's top.
    t_grasp = np.array(robot.fk([*q_grasp, GRIPPER_Q]))
    jaw_bottom = (t_grasp @ np.array([0.0, 0.0, -0.03, 1.0]))[:3]  # jaw half = 0.03
    top_face = np.array([carried["pos"][0], 0.0, carried["pos"][2] + carried["halfExtents"][2]])
    assert np.linalg.norm((jaw_bottom - top_face)[[0, 2]]) < 5e-3, (
        f"{path.name}: the witness grasp pose puts the jaw at {jaw_bottom.round(4)}, "
        f"not on {prop}'s top face {top_face.round(4)}"
    )

    # Carry it: the weld holds the prop rigid in the jaw frame.
    rel = np.linalg.inv(t_grasp) @ np.append(spawn[prop], 1.0)
    held = dict(spawn)
    held[prop] = tuple((np.array(robot.fk([*q_hold, GRIPPER_Q])) @ rel)[:3])

    at_rest = {name: (0.0, 0.0, 0.0) for name in spawn}
    predicate = clone(task.success)
    predicate.reset(SuccessState(prop_pos=spawn, prop_vel=at_rest))
    assert predicate(SuccessState(prop_pos=held, prop_vel=at_rest)), (
        f"{path.name} is UNSATISFIABLE at its witness: carrying {prop} to "
        f"{np.round(held[prop], 4).tolist()} does not satisfy "
        f"{task.success.describe()}"
    )
