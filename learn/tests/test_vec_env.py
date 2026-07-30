"""VecSimEnv substrate tests: shapes, determinism, auto-reset, the one example
task (reach_task), the fps/timestep cadence warning, rollout_random stream
independence, structured props / q0 / from_task, and the image-observation
smoke. CPU-only, seconds."""

import pathlib
import warnings

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
caliper = pytest.importorskip("caliper")

from caliper_learn.collect import _resolve_urdf  # noqa: E402
from caliper_learn.task import load_task  # noqa: E402
from caliper_learn.vec_env import VecSimEnv, reach_task, rollout_random  # noqa: E402

TASKS = pathlib.Path(__file__).resolve().parents[2] / "oracle" / "fixtures" / "tasks"


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


def test_fps_not_dividing_timestep_warns(robot):
    """Regression: fps=60 with timestep=1e-3 rounds to 17 substeps — a 17 ms
    control period sold as 16.67 ms, so sim time silently drifted against
    anything timestamped at fps (invisible to P005). Must warn loudly."""
    with pytest.warns(UserWarning, match="effective rate"):
        VecSimEnv(robot, 1, fps=60, timestep=1e-3)
    # an exact divisor stays silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        VecSimEnv(robot, 1, fps=50, timestep=1e-3)


def test_rollout_random_rng_does_not_alias_env_stream(robot):
    """Regression: the action RNG was default_rng(seed) — byte-identical to
    env 0's post-reset stream (reset(seed=seed) reseeds env i to
    default_rng(seed + i)), so the 'random' actions were a deterministic
    replay of the stream that also drives env 0's reset jitter and
    randomization draws. The action seed must be offset past every env."""
    n = robot.ndof
    env = VecSimEnv(robot, 2, fps=50, seed=0)
    out = rollout_random(env, 1, seed=0)
    b = env.action_bounds()
    # what the old aliased stream would have produced as the first action batch
    aliased = np.random.default_rng(0).uniform(b[:, 0], b[:, 1], size=(2, n))
    assert not np.array_equal(out["actions"][0], aliased)
    # the offset stream is the pinned new behavior (recorded expectation)
    expected = np.random.default_rng(0 + env.num_envs).uniform(
        b[:, 0], b[:, 1], size=(2, n)
    )
    assert np.array_equal(out["actions"][0], expected)


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


# ----- structured props / q0 / from_task --------------------------------------


CUBE = {
    "name": "cube",
    "kind": "box",
    "halfExtents": [0.025, 0.025, 0.025],
    "pos": [0.3, 0.0, 0.3],
    "mass": 0.05,
}


def test_props_add_a_free_body_the_robot_dofs_still_lead(robot):
    """The structured prop path: the engine builds the body, the robot keeps the
    qpos prefix, and the prop is addressable BY NAME through success_state()."""
    n = robot.ndof
    with VecSimEnv(robot, 2, fps=50, seed=0, ground=0.0, props=[CUBE]) as env:
        assert env.model.nq == n + 7 and env.model.nv == n + 6
        obs = env.reset(seed=0)
        # Observations still cover the ROBOT only.
        assert obs["state"].shape == (2, 2 * n)
        state = env.success_state(0)
        # Addressable under both spellings the exporter produces.
        assert state.pos("cube") == pytest.approx(CUBE["pos"])
        assert state.pos("prop_cube") == pytest.approx(CUBE["pos"])
        assert state.vel("cube") == pytest.approx([0.0, 0.0, 0.0])
        # Dropped from 0.3 m, it falls: props are simulated, not decoration.
        for _ in range(20):
            env.step(obs["state"][:, :n].astype(np.float64))
        assert env.success_state(0).pos("cube")[2] < CUBE["pos"][2]
        # ... and each env has its own copy of it.
        assert env.success_state(1).pos("cube")[2] < CUBE["pos"][2]


def test_props_and_extra_xml_compose(robot):
    with VecSimEnv(
        robot,
        1,
        fps=50,
        seed=0,
        extra_xml='<camera name="ots" pos="1 -1 1"/>',
        props=[CUBE],
    ) as env:
        assert env.model.nq == robot.ndof + 7
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "ots") >= 0
        # Joint-name resolution, not prefix arithmetic: still the robot's block.
        obs = env.reset(seed=0)
        assert obs["state"].shape == (1, 2 * robot.ndof)


def test_q0_moves_the_reset_center_and_stays_in_limits(robot):
    n = robot.ndof
    lims = robot.joint_limits
    q0 = [0.0 if lim is None else lim[1] for lim in lims]  # every joint AT its upper limit
    with VecSimEnv(robot, 4, fps=50, seed=0, q0=q0, init_jitter=0.5) as env:
        q = env.reset(seed=0)["state"][:, :n]
        bounds = env.action_bounds()
        # Jitter around an at-limit pose is clipped back inside the limits...
        assert np.all(q >= bounds[:, 0] - 1e-12) and np.all(q <= bounds[:, 1] + 1e-12)
        # ... and sits in the upper half, not around the midpoint.
        assert np.all(q.mean(axis=0) > env._mid)

    # Without q0 the sampling is bit-identical to the pre-q0 behavior.
    a = VecSimEnv(robot, 2, fps=50, seed=0).reset(seed=0)["state"]
    b = VecSimEnv(robot, 2, fps=50, seed=0, q0=None).reset(seed=0)["state"]
    assert np.array_equal(a, b)

    with pytest.raises(ValueError, match="robot has"):
        VecSimEnv(robot, 1, q0=[0.0] * (n + 1))
    with pytest.raises(ValueError, match="q0 must be finite"):
        VecSimEnv(robot, 1, q0=[float("nan")] * n)
    bounded = next((i for i, lim in enumerate(lims) if lim is not None), None)
    if bounded is not None:
        bad = [0.0 if lim is None else lim[0] for lim in lims]
        bad[bounded] = lims[bounded][0] - 1.0
        with pytest.raises(ValueError, match="outside joint"):
            VecSimEnv(robot, 1, q0=bad)


def test_structured_props_survive_model_randomization(robot):
    """The MJCF rebuild keeps the prop, and `mass_scale` scales the ROBOT's
    bodies only — a randomized arm must not silently randomize the payload it is
    being evaluated on."""
    from caliper_learn.randomize import RandomizationSpec

    spec = RandomizationSpec(mass_scale=(0.5, 0.6), joint_damping=(0.0, 0.1))
    with VecSimEnv(robot, 2, fps=50, seed=0, ground=0.0, props=[CUBE], randomization=spec) as env:
        env.reset(seed=0)
        assert [m.nq for m in env._models] == [robot.ndof + 7] * 2  # prop kept
        for m in env._models:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "prop_cube")
            assert m.body_mass[bid] == pytest.approx(CUBE["mass"])


def test_from_task_wires_the_whole_artifact():
    """One file in, a scored env out: scene, verdict, start pose, rate, budget."""
    task = load_task(TASKS / "pick_cube.caliper-task.json")
    with VecSimEnv.from_task(task, 2) as env:
        assert env.ndof == 3 and env.fps == task.fps
        assert env.model.nq == env.ndof + 7  # the task's one prop
        assert env._max_episode_steps == task.steps(task.fps) == 1000
        assert env.success_predicate is task.success
        obs = env.reset(seed=0)
        # q0 = [0, 0, 0.02] with the default 0.2 jitter: near the start pose,
        # nowhere near the joint midpoints for the two ±1.5 rad joints.
        assert np.all(np.abs(obs["state"][:, :2]) < 0.31)
        state = env.success_state(0)
        assert state.pos("cube") == pytest.approx(task.props[0]["pos"])
        # The predicate is live and cloned per env (no shared baseline).
        _o, _r, _te, _tr, info = env.step(obs["state"][:, :3].astype(np.float64))
        assert info["success"].shape == (2,) and not info["success"].any()

    # An already-loaded robot is reused rather than re-parsed.
    r = caliper.Robot.from_urdf(str(task.robot_path))
    with VecSimEnv.from_task(task, 1, robot=r, fps=25) as env:
        assert env.robot is r and env.fps == 25
        assert env._max_episode_steps == task.steps(25) == 500


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


def test_obs_images_render_the_props_too(robot):
    """The render scene must BE the scene: a prop the policy is meant to grasp
    cannot be missing from the images it trains on. The scene's qpos layout has
    to match the env's for `render(full qpos)` to be accepted at all."""
    n = robot.ndof
    with VecSimEnv(
        robot, 1, fps=50, obs_images=True, image_size=(64, 64), seed=0,
        ground=0.0, props=[CUBE],
    ) as env:
        assert env._scenes[0].model.nq == env.model.nq
        obs = env.reset(seed=0)
        assert obs["image"].shape == (1, 64, 64, 3)
        # Move the prop far out of frame and the pixels must change.
        before = obs["image"].copy()
        env._data[0].qpos[n : n + 3] = [5.0, 5.0, 5.0]
        mujoco.mj_forward(env.model, env._data[0])
        after = env._obs()["image"]
        assert not np.array_equal(before, after)
