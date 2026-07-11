"""VecSimEnv substrate tests: shapes, determinism, auto-reset, the one example
task (reach_task), and the image-observation smoke. CPU-only, seconds."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
caliper = pytest.importorskip("caliper")

from caliper_learn.collect import _resolve_urdf  # noqa: E402
from caliper_learn.vec_env import VecSimEnv, reach_task, rollout_random  # noqa: E402


@pytest.fixture(scope="module")
def robot():
    # collide_arm: the standard planner fixture (has inertials -> MJCF-exportable)
    return caliper.Robot.from_urdf(_resolve_urdf("planner", None))


def _moving_frame(robot) -> str:
    """Pick the frame whose world position moves most between two configs, so
    the reach reward actually depends on q."""
    n = robot.ndof
    qa, qb = [0.0] * n, [0.4] * n

    def pos(q, f):
        p = robot.fk(q, f)
        return np.array([p[0][3], p[1][3], p[2][3]])

    return max(robot.frame_names(), key=lambda f: np.linalg.norm(pos(qa, f) - pos(qb, f)))


def test_shapes_dtypes_and_rollout(robot):
    n = robot.ndof
    env = VecSimEnv(robot, 3, fps=50, seed=1)
    obs = env.reset()
    assert set(obs) == {"state"}
    assert obs["state"].shape == (3, 2 * n) and obs["state"].dtype == np.float32
    # velocities start at zero after reset
    assert np.all(obs["state"][:, n:] == 0.0)

    acts = obs["state"][:, :n].astype(np.float64)  # hold current pose
    obs2, r, te, tr, info = env.step(acts)
    assert obs2["state"].shape == (3, 2 * n) and obs2["state"].dtype == np.float32
    assert r.shape == (3,) and r.dtype == np.float64
    assert np.all(r == 0.0)  # default task: zero reward
    assert te.shape == (3,) and te.dtype == np.bool_ and not te.any()
    assert tr.shape == (3,) and tr.dtype == np.bool_ and not tr.any()
    assert not info["reset_mask"].any()

    out = rollout_random(env, 3, seed=0)
    assert out["states"].shape == (3, 3, 2 * n)
    assert out["actions"].shape == (3, 3, n)
    assert out["rewards"].shape == (3, 3)
    assert out["terminated"].shape == (3, 3) and out["truncated"].shape == (3, 3)

    with pytest.raises(ValueError):
        env.step(np.zeros((3, n + 1)))
    with pytest.raises(ValueError):
        VecSimEnv(robot, 1, ctrl_mode="servo")


def test_determinism_same_seed_identical_trajectories(robot):
    n = robot.ndof
    rng = np.random.default_rng(3)
    action_seq = rng.uniform(-0.5, 0.5, size=(20, 2, n))

    def run():
        env = VecSimEnv(robot, 2, fps=50, seed=7)
        states = [env.reset()["state"]]
        for a in action_seq:
            obs, *_ = env.step(a)
            states.append(obs["state"])
        return np.stack(states)

    a, b = run(), run()
    assert np.array_equal(a, b)  # bitwise, not allclose


def test_autoreset_fires_and_is_flagged(robot):
    n = robot.ndof
    env = VecSimEnv(robot, 2, fps=50, seed=0)
    env.set_task(None, lambda qpos, qvel, i: True)  # terminate every step
    obs = env.reset(seed=0)
    acts = obs["state"][:, :n].astype(np.float64)

    obs2, _r, te, tr, info = env.step(acts)
    assert te.all() and not tr.any()
    assert info["reset_mask"].all()
    finals = info["final_observation"]
    assert len(finals) == 2
    for i, f in enumerate(finals):
        assert f is not None and f.shape == (2 * n,) and f.dtype == np.float32
        # the returned obs is the RESET one, not the terminal one
        assert not np.array_equal(obs2["state"][i], f)
    # the env keeps stepping fine after auto-reset
    _obs3, _r, te2, _tr, info2 = env.step(acts)
    assert te2.all() and info2["reset_mask"].all()


def test_reach_task_reward_increases(robot):
    n = robot.ndof
    frame = _moving_frame(robot)
    q_goal = np.zeros(n)
    pose = robot.fk(list(q_goal), frame)
    target = [pose[0][3], pose[1][3], pose[2][3]]

    reward_fn, _term = reach_task(robot, frame, target, tol=0.05)
    env = VecSimEnv(robot, 1, fps=50, seed=5, init_jitter=0.3)
    env.set_task(reward_fn)  # reward only: no termination, no auto-reset
    obs = env.reset()
    rewards = []
    for _ in range(60):
        q = obs["state"][0, :n].astype(np.float64)
        a = q + 0.5 * (q_goal - q)  # crude proportional drive toward the goal
        obs, r, _te, _tr, _info = env.step(a[None, :])
        rewards.append(float(r[0]))
    assert rewards[0] < 0.0  # started away from the target
    assert rewards[-1] > rewards[0]  # got closer -> negative-distance reward rose
    assert abs(rewards[-1]) < 0.7 * abs(rewards[0])  # closed a real fraction of the gap


def test_obs_images_smoke(robot):
    n = robot.ndof
    with VecSimEnv(robot, 2, fps=50, obs_images=True, image_size=(64, 64), seed=0) as env:
        obs = env.reset()
        assert set(obs) == {"state", "image"}
        assert obs["image"].shape == (2, 64, 64, 3) and obs["image"].dtype == np.uint8
        acts = obs["state"][:, :n].astype(np.float64)
        obs2, *_ = env.step(acts)
        assert obs2["image"].shape == (2, 64, 64, 3)
        # the two envs render independently but from identical state distributions'
        # own draws — just require non-degenerate pixels (an actual scene, not black)
        assert obs["image"].max() > 0
