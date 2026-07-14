"""Domain randomization + coverage generator tests: exact draw determinism,
parse-compare MJCF edits (targeted fields change, NOTHING else), env-level
randomization actually changing the dynamics, and the doctor→generator loop
measurably raising bin occupancy on a hole-ridden dataset. CPU-only, seconds
(the lerobot load check skips where lerobot is not installed)."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
caliper = pytest.importorskip("caliper")

from caliper_learn.collect import _bounds, _resolve_urdf  # noqa: E402
from caliper_learn.coverage_gen import generate_coverage  # noqa: E402
from caliper_learn.randomize import (  # noqa: E402
    ENV_KEYS,
    MODEL_KEYS,
    RandomizationSpec,
    apply_to_env,
    apply_to_mjcf,
    sample,
)
from caliper_learn.vec_env import VecSimEnv  # noqa: E402

FULL_SPEC = RandomizationSpec(
    mass_scale=(0.8, 1.2),
    joint_damping=(0.0, 0.4),
    joint_frictionloss=(0.0, 0.1),
    kp_scale=(0.7, 1.3),
    kd_scale=(0.7, 1.3),
    camera_pos_jitter=(-0.05, 0.05),
    camera_euler_jitter=(-0.1, 0.1),
    spawn_jitter=(-0.05, 0.05),
    gravity_jitter=(-0.3, 0.3),
)
MODEL_SPEC = RandomizationSpec(
    mass_scale=(0.8, 1.2), joint_damping=(0.05, 0.4), gravity_jitter=(-0.3, 0.3)
)


@pytest.fixture(scope="module")
def robot():
    # collide_arm: the standard planner fixture (has inertials -> MJCF-exportable)
    return caliper.Robot.from_urdf(_resolve_urdf("planner", None))


# ----- sample: seeding + serialization ------------------------------------------


def test_draw_determinism_exact(robot):
    n = robot.ndof
    a = sample(FULL_SPEC, 7, n)
    b = sample(FULL_SPEC, 7, n)
    assert a == b  # exact equality, not allclose — the seeding contract
    assert sample(FULL_SPEC, 8, n) != a  # a different seed actually differs
    # every spec'd field appears, with the right per-field length
    assert set(a) == MODEL_KEYS | ENV_KEYS
    assert len(a["mass_scale"]) == len(a["spawn_offset"]) == n
    assert len(a["gravity"]) == len(a["camera_pos"]) == 3
    assert isinstance(a["kp_scale"], float)


def test_draw_is_json_diffable(robot):
    a = sample(FULL_SPEC, 0, robot.ndof)
    assert json.loads(json.dumps(a)) == a  # plain floats/lists round-trip losslessly


def test_draw_stream_stable_across_disabled_fields(robot):
    """Disabling a field must not reshuffle the remaining fields' values."""
    n = robot.ndof
    full = sample(FULL_SPEC, 3, n)
    partial = sample(RandomizationSpec(mass_scale=(0.8, 1.2)), 3, n)
    assert partial == {"mass_scale": full["mass_scale"]}


def test_spec_validation_rejects_bad_ranges():
    with pytest.raises(ValueError):
        RandomizationSpec(mass_scale=(1.2, 0.8))  # low > high
    with pytest.raises(ValueError):
        RandomizationSpec(kp_scale=(0.0, 1.5))  # multiplier low must be > 0
    with pytest.raises(ValueError):
        RandomizationSpec(joint_damping=(-0.1, 0.4))  # damping must be >= 0
    with pytest.raises(ValueError):
        RandomizationSpec(gravity_jitter=(0.0, float("nan")))
    with pytest.raises(ValueError):
        sample(FULL_SPEC, 0, 0)  # ndof must be >= 1


# ----- apply_to_mjcf: parse-compare ---------------------------------------------


def _canon(elem):
    """Element -> nested comparable tuple (tag, sorted attrs, children)."""
    return (elem.tag, tuple(sorted(elem.attrib.items())), tuple(_canon(c) for c in elem))


def _robot_bodies(root):
    return [b for b in root.iter("body") if b.get("name", "").startswith("b_")]


def test_apply_to_mjcf_changes_exactly_the_targeted_fields(robot):
    n = robot.ndof
    xml = caliper.model_to_mjcf(robot)
    draw = sample(MODEL_SPEC, 5, n)
    out = apply_to_mjcf(draw, xml)

    base, edited = ET.fromstring(xml), ET.fromstring(out)
    # gravity: base + drawn offset, exactly
    g0 = [float(v) for v in base.find("option").get("gravity").split()]
    g1 = [float(v) for v in edited.find("option").get("gravity").split()]
    assert g1 == [b + o for b, o in zip(g0, draw["gravity"])]

    bb, eb = _robot_bodies(base), _robot_bodies(edited)
    assert len(bb) == len(eb) == n
    for i, (b0, b1) in enumerate(zip(bb, eb)):
        i0, i1 = b0.find("inertial"), b1.find("inertial")
        s = draw["mass_scale"][i]
        assert float(i1.get("mass")) == float(i0.get("mass")) * s
        # inertia scales WITH mass (fixed geometry) — all 6 components
        fi0 = [float(v) for v in i0.get("fullinertia").split()]
        fi1 = [float(v) for v in i1.get("fullinertia").split()]
        assert fi1 == [v * s for v in fi0]
        assert float(b1.find("joint").get("damping")) == draw["joint_damping"][i]

    # ... and NOTHING else: normalize the targeted attrs on both trees, then
    # the documents must be structurally identical.
    for tree in (base, edited):
        tree.find("option").set("gravity", "X")
        for b in _robot_bodies(tree):
            inertial, joint = b.find("inertial"), b.find("joint")
            inertial.set("mass", "X")
            inertial.set("fullinertia", "X")
            joint.attrib.pop("damping", None)
    assert _canon(base) == _canon(edited)

    # the edited document still compiles, with the same tree shape
    m = mujoco.MjModel.from_xml_string(out)
    assert m.nq == n


def test_apply_to_mjcf_negative_paths(robot):
    xml = caliper.model_to_mjcf(robot)
    # empty draw -> structurally identical document (a no-op is a no-op)
    assert _canon(ET.fromstring(apply_to_mjcf({}, xml))) == _canon(ET.fromstring(xml))
    # env-level keys are ignored here, never smuggled into the model
    out = apply_to_mjcf({"kp_scale": 2.0}, xml)
    assert _canon(ET.fromstring(out)) == _canon(ET.fromstring(xml))
    with pytest.raises(ValueError):
        apply_to_mjcf({"masss_scale": [1.0]}, xml)  # typo'd key fails loudly
    with pytest.raises(ValueError):
        apply_to_mjcf({"mass_scale": [1.0]}, xml)  # wrong per-joint length
    with pytest.raises(ValueError):
        apply_to_mjcf({"gravity": [0.1, 0.2]}, xml)  # gravity needs 3 components


# ----- env-level randomization ---------------------------------------------------


def _roll(robot, seed, steps=8, spec=FULL_SPEC):
    """Fixed hold-at-midpoint targets so ONLY the randomization varies."""
    n = robot.ndof
    env = VecSimEnv(
        robot, 2, fps=50, seed=seed,
        randomization=RandomizationSpec(**{
            f: getattr(spec, f) for f in (
                "mass_scale", "joint_damping", "kp_scale", "kd_scale",
                "spawn_jitter", "gravity_jitter",
            )
        }),
    )
    env.reset()
    draws = env.randomization_draws
    acts = np.tile(env._mid, (2, 1))
    states, info = [], {}
    for _ in range(steps):
        obs, _r, _te, _tr, info = env.step(acts)
        states.append(obs["state"])
    return np.stack(states), draws, info


def test_env_randomization_seeded(robot):
    s1, d1, info1 = _roll(robot, seed=11)
    s2, d2, _ = _roll(robot, seed=11)
    s3, d3, _ = _roll(robot, seed=12)
    assert np.array_equal(s1, s2) and d1 == d2  # same seed: bitwise identical
    assert not np.array_equal(s1, s3) and d1 != d3  # different seed: differs
    assert d1[0] != d1[1]  # per-env streams draw independently
    # draws ride along in step info, index-aligned
    assert info1["randomization"] == d1


def test_env_randomization_changes_dynamics(robot):
    """Positive control: the same seed WITH vs WITHOUT randomization must
    produce different trajectories — otherwise the draws are decorative."""
    n = robot.ndof
    plain = VecSimEnv(robot, 2, fps=50, seed=11)
    plain.reset()
    acts = np.tile(plain._mid, (2, 1))
    states = []
    for _ in range(8):
        obs, *_ = plain.step(acts)
        states.append(obs["state"])
    randomized, _, _ = _roll(robot, seed=11)
    assert not np.array_equal(np.stack(states), randomized)
    # no spec -> no info key and no draws (the substrate stays unchanged)
    _obs, _r, _te, _tr, info = plain.step(acts)
    assert "randomization" not in info
    assert plain.randomization_draws == [None, None]


def test_env_model_rebuild_is_per_env(robot):
    env = VecSimEnv(robot, 2, fps=50, seed=0, randomization=MODEL_SPEC)
    env.reset()
    # model-level draws give each env its OWN compiled model (documented cost)
    assert env._models[0] is not env._models[1]
    assert env._models[0] is not env.model
    # the randomized masses actually landed in the compiled models
    m0 = env._models[0].body_mass.sum()
    m1 = env._models[1].body_mass.sum()
    assert m0 != m1 != env.model.body_mass.sum()


def test_camera_jitter_moves_the_scene_camera(robot):
    spec = RandomizationSpec(camera_pos_jitter=(0.02, 0.08), camera_euler_jitter=(-0.1, 0.1))
    with VecSimEnv(
        robot, 2, fps=50, seed=0, obs_images=True, image_size=(32, 32),
        randomization=spec,
    ) as env:
        env.reset()
        cid = mujoco.mj_name2id(
            env._scenes[0].model, mujoco.mjtObj.mjOBJ_CAMERA, env._scenes[0].camera
        )
        p0 = env._scenes[0].model.cam_pos[cid].copy()
        p1 = env._scenes[1].model.cam_pos[cid].copy()
        base = env._scenes[0]._rand_base_cam[0]
        assert not np.array_equal(p0, p1)  # per-env draws -> different cameras
        assert not np.array_equal(p0, base)  # and both moved off the base pose
        # jitter is around the ORIGINAL pose: repeated resets never accumulate
        env.reset(seed=0)
        first = env._scenes[0].model.cam_pos[cid].copy()
        env.reset(seed=0)
        assert np.array_equal(env._scenes[0].model.cam_pos[cid], first)


def test_apply_to_env_rejects_bad_input(robot):
    env = VecSimEnv(robot, 1, fps=50, seed=0)
    with pytest.raises(ValueError):
        apply_to_env({"bogus_key": 1.0}, env, index=0)
    with pytest.raises(ValueError):
        apply_to_env({"kp_scale": 2.0}, env, index=5)  # index out of range
    with pytest.raises(ValueError):
        apply_to_env({"spawn_offset": [0.1]}, env, index=0)  # wrong dof count


# ----- the doctor→generator loop -------------------------------------------------


def _holey_dataset(robot, out, n_episodes=4, frames=40):
    """A deliberately hole-ridden v3.0 dataset: episodes alternate between two
    tiny clusters near ±0.4 of each joint's half-range, never crossing the
    middle. The doctor bins between OBSERVED min and max, so the wide-but-empty
    gap between the clusters is exactly the D007 shape (a single narrow band
    would trivially cover its own span and report full occupancy)."""
    b = _bounds(robot)
    mid, half = b.mean(axis=1), 0.5 * (b[:, 1] - b[:, 0])
    rec = caliper.RecorderV3(robot, str(out), fps=50)
    for ep in range(n_episodes):
        rng = np.random.default_rng(ep)
        phase = rng.uniform(0.0, 2 * np.pi, size=robot.ndof)
        center = mid + (0.4 if ep % 2 else -0.4) * half
        rec.start_episode(f"cluster {ep % 2} episode {ep}")
        for k in range(frames):
            q = center + 0.03 * half * np.sin(0.3 * k + phase)
            q2 = center + 0.03 * half * np.sin(0.3 * (k + 1) + phase)
            rec.append([float(v) for v in q], [float(v) for v in q2], k / 50)
        rec.finalize_episode()
    return rec.close()


def test_coverage_gen_raises_min_occupancy(robot, tmp_path):
    src = _holey_dataset(robot, tmp_path / "narrow")
    rep = generate_coverage(src, robot, str(tmp_path / "filled"), episodes=3, seed=0)

    assert rep.episodes_replayed == 4 and rep.episodes_added > 0
    # THE closed-loop assertion: the doctor's worst-dof occupancy went up
    assert rep.min_occupancy_after > rep.min_occupancy_before
    assert all(a >= 0 for a in rep.occupancy_after)
    assert rep.d007_after <= rep.d007_before

    # the input was never mutated: same episode count, same first-episode bytes
    rd_src = caliper.DatasetReaderV3.open(src)
    assert rd_src.total_episodes == 4
    assert rd_src.read_episode(0) == caliper.DatasetReaderV3.open(src).read_episode(0)
    # the output holds replay + additions, and replay is verbatim
    rd_out = caliper.DatasetReaderV3.open(rep.out_root)
    assert rd_out.total_episodes == 4 + rep.episodes_added
    assert rd_out.read_episode(0) == rd_src.read_episode(0)
    # report serializes deterministically (sorted keys)
    assert json.loads(rep.to_json())["out_root"] == rep.out_root
    assert "min bin occupancy" in rep.render_text()


def test_coverage_gen_deterministic_reruns(robot, tmp_path):
    src = _holey_dataset(robot, tmp_path / "narrow")
    r1 = generate_coverage(src, robot, str(tmp_path / "a"), episodes=2, seed=0)
    r2 = generate_coverage(src, robot, str(tmp_path / "b"), episodes=2, seed=0)
    assert r1.occupancy_after == r2.occupancy_after
    rd1 = caliper.DatasetReaderV3.open(r1.out_root)
    rd2 = caliper.DatasetReaderV3.open(r2.out_root)
    assert rd1.total_episodes == rd2.total_episodes
    for ep in range(rd1.total_episodes):
        assert rd1.read_episode(ep) == rd2.read_episode(ep)


def test_coverage_gen_negative_paths(robot, tmp_path):
    src = _holey_dataset(robot, tmp_path / "narrow")
    with pytest.raises(ValueError):
        generate_coverage(src, robot, str(tmp_path / "x"), episodes=0)
    with pytest.raises(ValueError):
        generate_coverage(src, robot, str(tmp_path / "y"), bins=1)
    # an unreadable input fails loudly at the doctor, before anything is written
    with pytest.raises(ValueError):
        generate_coverage(tmp_path / "does-not-exist", robot, str(tmp_path / "z"))
    assert not (tmp_path / "z").exists()


def test_coverage_output_passes_lerobot_load(robot, tmp_path):
    pytest.importorskip("lerobot", reason="lerobot not installed")
    lerobot_dataset_mod = pytest.importorskip("lerobot.datasets.lerobot_dataset")

    src = _holey_dataset(robot, tmp_path / "narrow")
    rep = generate_coverage(src, robot, str(tmp_path / "filled"), episodes=2, seed=0)
    ds = lerobot_dataset_mod.LeRobotDataset("caliper/coverage", root=rep.out_root)
    assert ds.num_episodes == 4 + rep.episodes_added
    frame = ds[0]
    assert frame["observation.state"].shape == (robot.ndof,)
    assert frame["action"].shape == (robot.ndof,)
