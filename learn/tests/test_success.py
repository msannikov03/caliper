"""Success-predicate tests: the predicate math (initial vs absolute reference,
inclusive zone faces, settled speed, combinators, exact JSON round-trips,
stable describe() sentences), the VecSimEnv wiring (props reach the env only
through extra_xml, info["success"]/["final_success"] follow the
final_observation convention, the no-predicate path is untouched, and a prop
in the scene does not disturb the arm), and predicate-sourced eval (hand-
counted rate, Wilson bounds identical to the same counts, the criterion
sentence carried into the report and the autopsy verdict). CPU-only, seconds.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
caliper = pytest.importorskip("caliper")

from caliper_learn.autopsy import _verdict  # noqa: E402
from caliper_learn.collect import _bounds, _resolve_urdf  # noqa: E402
from caliper_learn.eval import (  # noqa: E402
    ALL_EPISODES_FAILED,
    EvalConfig,
    evaluate,
    reach_eval_task,
    render_text,
    to_json,
    wilson_interval,
)
from caliper_learn.success import (  # noqa: E402
    AllOf,
    AnyOf,
    Lifted,
    PlacedInZone,
    SuccessState,
    Zone,
    as_predicate,
    clone,
    from_dict,
    state_from_mujoco,
)
from caliper_learn.vec_env import VecSimEnv  # noqa: E402

# A free-floating 4 cm cube. Props have no structured Python path today
# (`caliper.model_to_mjcf` takes no `props=`), so they enter the scene as
# verbatim MJCF — the body name follows the exporter's `prop_<name>`.
CUBE_AT = (
    '<body name="prop_cube" pos="{x} {y} {z}">'
    '<freejoint name="prop_cube_free"/>'
    '<inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>'
    '<geom type="box" size="0.02 0.02 0.02"/>'
    "</body>"
)


@pytest.fixture(scope="module")
def robot():
    # collide_arm: the standard planner fixture (has inertials -> MJCF-exportable)
    return caliper.Robot.from_urdf(_resolve_urdf("planner", None))


def _state(z, *, prop="cube", vel=None, xy=(0.0, 0.0)):
    """A hand-built SuccessState — the predicates never need a simulator."""
    return SuccessState(
        prop_pos={prop: (xy[0], xy[1], z)},
        prop_vel=None if vel is None else {prop: vel},
    )


# ----- Lifted -------------------------------------------------------------------


def test_lifted_initial_reference_captures_on_reset():
    # Heights here are binary-exact (eighths): the comparison is a plain `>=`,
    # so "exactly the threshold" only means anything for values floats can
    # actually hold — 0.15 - 0.10 is 0.049999999999999996, not 0.05.
    p = Lifted("cube", 0.125)
    p.reset(_state(0.25))  # this episode's floor is z = 0.25
    assert p(_state(0.3125)) is False  # 0.0625 m up: not yet
    assert p(_state(0.375)) is True  # exactly the threshold counts
    assert p(_state(0.75)) is True
    # A new episode from a HIGHER start re-captures: the same absolute height
    # that succeeded above is now a non-lift.
    p.reset(_state(0.625))
    assert p(_state(0.6875)) is False
    # reset() with no state clears instead: the next call becomes the reference.
    p.reset()
    assert p(_state(0.5)) is False  # captured here
    assert p(_state(0.625)) is True


def test_lifted_absolute_reference_ignores_the_start():
    p = Lifted("cube", 0.20, ref="absolute")
    p.reset(_state(0.50))  # start above the bar: already successful
    assert p(_state(0.50)) is True
    assert p(_state(0.20)) is True  # inclusive
    assert p(_state(0.19)) is False
    # absolute never captures anything, so a reset changes nothing
    p.reset()
    assert p(_state(0.19)) is False


def test_lifted_validation_is_loud():
    with pytest.raises(ValueError):
        Lifted("", 0.05)
    with pytest.raises(ValueError):
        Lifted("cube", float("nan"))
    with pytest.raises(ValueError):
        Lifted("cube", 0.05, ref="sideways")
    with pytest.raises(ValueError):
        Lifted("cube", 0.0)  # a lift of zero is not a lift
    with pytest.raises(ValueError):
        Lifted("cube", -0.05)
    # a negative ABSOLUTE threshold is legal (below the world origin)
    assert Lifted("cube", -0.05, ref="absolute")(_state(-0.04)) is True
    # an unknown prop names what IS there
    with pytest.raises(ValueError, match="known props"):
        Lifted("brick", 0.05)(_state(0.1))


# ----- zones / PlacedInZone --------------------------------------------------------


def test_zone_faces_are_inclusive():
    # Binary-exact eighths again, so "on the face" is a fact and not a rounding
    # accident (0.4 - 0.3 is 0.10000000000000003).
    z = Zone(center=(0.25, 0.0, 0.25), half=(0.125, 0.125, 0.125))
    assert z.contains((0.25, 0.0, 0.25))
    for corner in ((0.375, 0.125, 0.375), (0.125, -0.125, 0.125)):  # exact corners
        assert z.contains(corner)
    assert not z.contains((0.375 + 1e-9, 0.0, 0.25))
    assert not z.contains((0.25, 0.0, 0.5))
    # a zero half-extent is a legal (degenerate) plane constraint
    assert Zone((0, 0, 0), (1, 1, 0)).contains((0.5, 0.5, 0.0))
    with pytest.raises(ValueError):
        Zone((0, 0, 0), (1, 1, -1))
    with pytest.raises(ValueError):
        Zone((0, 0), (1, 1, 1))
    with pytest.raises(ValueError):
        Zone((0, 0, float("inf")), (1, 1, 1))


def test_placed_in_zone_with_and_without_settling():
    zone = Zone(center=(0.0, 0.0, 0.02), half=(0.05, 0.05, 0.05))
    loose = PlacedInZone("cube", zone)
    settled = PlacedInZone("cube", zone, settled_speed=0.01)

    inside_fast = _state(0.02, vel=(0.0, 0.3, 0.0))
    inside_slow = _state(0.02, vel=(0.0, 0.005, 0.0))
    outside = _state(0.50, vel=(0.0, 0.0, 0.0))

    assert loose(inside_fast) is True  # in the box, still flying through
    assert settled(inside_fast) is False  # ... which is not "placed"
    assert settled(inside_slow) is True
    assert loose(outside) is False and settled(outside) is False
    # speed is the full 3-vector norm, not per axis
    assert settled(_state(0.02, vel=(0.008, 0.008, 0.0))) is False

    # A settled check against a velocity-less state must NOT pass silently.
    with pytest.raises(ValueError, match="no velocities"):
        settled(SuccessState(prop_pos={"cube": (0.0, 0.0, 0.02)}))
    assert loose(SuccessState(prop_pos={"cube": (0.0, 0.0, 0.02)})) is True

    with pytest.raises(TypeError):
        PlacedInZone("cube", {"center": (0, 0, 0), "half": (1, 1, 1)})
    with pytest.raises(ValueError):
        PlacedInZone("cube", zone, settled_speed=-1.0)


# ----- combinators -------------------------------------------------------------------


def test_combinators_and_their_evaluation_order():
    lift = Lifted("cube", 0.125)
    zone = PlacedInZone("cube", Zone((0.0, 0.0, 0.5), (0.125, 0.125, 0.125)))
    both, either = AllOf(lift, zone), AnyOf(lift, zone)

    both.reset(_state(0.25))
    assert both(_state(0.3)) is False  # neither
    assert both(_state(0.75)) is False  # lifted, but above the zone
    assert both(_state(0.5)) is True  # lifted AND inside
    either.reset(_state(0.25))
    assert either(_state(0.75)) is True
    assert either(_state(0.3)) is False

    # AllOf must not short-circuit: the SECOND term is a Lifted whose baseline
    # is captured on first evaluation. If a False first term skipped it, the
    # reference would be captured later and score a different episode.
    late = Lifted("cube", 0.125)
    guard = AllOf(PlacedInZone("cube", Zone((9.0, 9.0, 9.0), (0.1, 0.1, 0.1))), late)
    assert guard(_state(0.10)) is False  # first term False...
    assert late._z0 == pytest.approx(0.10)  # ...second term still captured

    with pytest.raises(ValueError):
        AllOf()
    with pytest.raises(TypeError):
        AnyOf("lifted")


# ----- serialization ----------------------------------------------------------------


def test_json_round_trip_is_exact():
    preds = [
        Lifted("cube", 0.05),
        Lifted("cube", 0.05, ref="absolute"),
        PlacedInZone("cube", Zone((0.3, -0.1, 0.02), (0.05, 0.05, 0.02))),
        PlacedInZone("cube", Zone((0.3, -0.1, 0.02), (0.05, 0.05, 0.02)), 0.01),
        AllOf(Lifted("cube", 0.05), PlacedInZone("cube", Zone((0, 0, 0), (1, 1, 1)))),
        AnyOf(AllOf(Lifted("a", 0.1)), Lifted("b", 0.2, ref="absolute")),
    ]
    for p in preds:
        d = p.to_dict()
        assert from_dict(d).to_dict() == d  # exact, not approximately
        assert from_dict(d) == p  # and the objects themselves
        assert from_dict(d).describe() == p.describe()

    import json

    d = preds[-2].to_dict()
    assert json.loads(json.dumps(d)) == d  # plain JSON all the way down
    assert d == {
        "kind": "all_of",
        "terms": [
            {"kind": "lifted", "prop": "cube", "height": 0.05, "ref": "initial"},
            {
                "kind": "placed_in_zone",
                "prop": "cube",
                "zone": {"center": [0.0, 0.0, 0.0], "half": [1.0, 1.0, 1.0]},
                "settled_speed": None,
            },
        ],
    }


def test_bad_specs_raise_instead_of_weakening_the_test():
    with pytest.raises(ValueError, match="unknown success predicate kind"):
        from_dict({"kind": "grasped", "prop": "cube"})
    with pytest.raises(ValueError, match="unknown key"):
        from_dict({"kind": "lifted", "prop": "cube", "height": 0.05, "hieght": 0.5})
    with pytest.raises(ValueError, match="unknown key"):
        from_dict(
            {"kind": "placed_in_zone", "prop": "c", "zone": {"center": [0, 0, 0], "half": [1, 1, 1], "z": 1}}
        )
    with pytest.raises(ValueError):
        from_dict({"kind": "placed_in_zone", "prop": "c", "zone": {"center": [0, 0, 0]}})
    with pytest.raises(TypeError):
        from_dict("lifted")


def test_describe_sentences_are_stable():
    assert Lifted("cube", 0.05).describe() == (
        "cube rises 0.050 m above its initial height"
    )
    assert Lifted("cube", 0.2, ref="absolute").describe() == (
        "cube's center reaches z >= 0.200 m"
    )
    assert PlacedInZone("cube", Zone((0.3, 0.0, 0.02), (0.05, 0.05, 0.02))).describe() == (
        "cube's center is inside the 0.100 x 0.100 x 0.040 m box centered at "
        "(0.300, 0.000, 0.020)"
    )
    assert PlacedInZone(
        "cube", Zone((0.3, 0.0, 0.02), (0.05, 0.05, 0.02)), settled_speed=0.01
    ).describe().endswith("and has come to rest (speed <= 0.010 m/s)")
    assert AllOf(Lifted("a", 0.1), Lifted("b", 0.2)).describe() == (
        "a rises 0.100 m above its initial height and "
        "b rises 0.200 m above its initial height"
    )
    assert " or " in AnyOf(Lifted("a", 0.1), Lifted("b", 0.2)).describe()
    assert Lifted("cube", 0.05).name == "lifted:cube"
    assert PlacedInZone("cube", Zone((0, 0, 0), (1, 1, 1))).name == "placed_in_zone:cube"
    assert AllOf(Lifted("a", 0.1), Lifted("b", 0.2)).name == "all_of(lifted:a, lifted:b)"


def test_as_predicate_and_clone():
    p = Lifted("cube", 0.125)
    assert as_predicate(p) is p
    assert as_predicate(p.to_dict()) == p
    assert as_predicate([p.to_dict()]) == p  # a 1-element sequence is not wrapped
    combined = as_predicate([p, PlacedInZone("cube", Zone((0, 0, 0), (1, 1, 1)))])
    assert isinstance(combined, AllOf) and len(combined.terms) == 2
    with pytest.raises(ValueError):
        as_predicate([])
    with pytest.raises(TypeError):
        as_predicate(42)

    # a clone shares NO captured baseline with its source
    p.reset(_state(0.25))
    c = clone(p)
    c.reset(_state(0.90))
    assert p(_state(0.375)) is True and c(_state(0.375)) is False


# ----- reading props out of a live sim -------------------------------------------------


def test_state_from_mujoco_keys_and_collisions(robot):
    with VecSimEnv(robot, 1, fps=50, seed=0, extra_xml=CUBE_AT.format(x=0.3, y=0.0, z=0.5)) as env:
        env.reset(seed=0)
        st = env.success_state(0)
        # addressable by the body name AND, for `prop_*` bodies, the bare name
        assert set(st.prop_pos) == {"prop_cube", "cube"}
        assert st.prop_pos["cube"] == st.prop_pos["prop_cube"]
        assert st.prop_pos["cube"] == pytest.approx((0.3, 0.0, 0.5))
        assert st.prop_vel["cube"] == pytest.approx((0.0, 0.0, 0.0))
        assert st.tip_pos is None  # VecSimEnv does not run FK for it
        with pytest.raises(ValueError):
            env.success_state(3)

    # `cube` and `prop_cube` in the same scene would both answer to "cube"
    clash = CUBE_AT.format(x=0.3, y=0.0, z=0.5) + CUBE_AT.format(
        x=0.0, y=0.3, z=0.5
    ).replace('"prop_cube"', '"cube"').replace('"prop_cube_free"', '"cube_free"')
    with VecSimEnv(robot, 1, fps=50, seed=0, extra_xml=clash) as env:
        with pytest.raises(ValueError, match="answer to the name"):
            state_from_mujoco(env.model, env._data[0])


# ----- VecSimEnv integration -------------------------------------------------------------


def test_no_predicate_leaves_step_info_untouched(robot):
    """Default success=None: not one new key, not one changed value."""
    with VecSimEnv(robot, 2, fps=50, seed=1) as env:
        obs = env.reset(seed=1)
        _obs, _r, _te, _tr, info = env.step(obs["state"][:, : env.ndof].astype(np.float64))
        assert set(info) == {"reset_mask"}
        assert env.success_predicate is None


def test_a_prop_in_the_scene_does_not_disturb_the_arm(robot):
    """The arm's trajectory must be bit-identical with and without a prop.

    This is the real test of the by-name dof resolution: `extra_xml` is
    injected BEFORE the robot bodies, so the cube's free joint takes qpos 0..6
    and the arm moves to 7... — index arithmetic on ndof would silently drive
    the cube's pose instead of the arm.
    """
    far_cube = CUBE_AT.format(x=5.0, y=5.0, z=5.0)  # nowhere near the arm
    rng = np.random.default_rng(7)
    with VecSimEnv(robot, 1, fps=50, seed=3) as plain, VecSimEnv(
        robot, 1, fps=50, seed=3, extra_xml=far_cube
    ) as propped:
        assert plain.model.nq == robot.ndof
        assert propped.model.nq == robot.ndof + 7  # one free joint
        assert propped._q_sel == slice(7, 7 + robot.ndof)
        a_plain, a_prop = plain.reset(seed=3), propped.reset(seed=3)
        assert np.array_equal(a_plain["state"], a_prop["state"])
        b = plain.action_bounds()
        for _ in range(20):
            act = rng.uniform(b[:, 0], b[:, 1], size=(1, robot.ndof))
            s1 = plain.step(act)[0]["state"]
            s2 = propped.step(act)[0]["state"]
            assert np.array_equal(s1, s2)  # exact, not approx


def test_success_flows_into_info_per_env(robot):
    """Two envs, one predicate spec: each env is scored against ITS own start."""
    cube = CUBE_AT.format(x=0.3, y=0.0, z=0.5)  # falls freely (no ground)
    pred = Lifted("cube", 0.20, ref="absolute")
    with VecSimEnv(robot, 2, fps=50, seed=0, extra_xml=cube, success=pred) as env:
        obs = env.reset(seed=0)
        hold = obs["state"][:, : env.ndof].astype(np.float64)
        _o, _r, _te, _tr, info = env.step(hold)
        assert info["success"].shape == (2,) and info["success"].dtype == np.bool_
        assert info["success"].all()  # z = 0.5 > 0.20, both envs
        assert "final_success" not in info  # nothing reset
        assert env.success_predicate is pred
        # the cube falls below the bar and stays there
        for _ in range(40):
            _o, _r, _te, _tr, info = env.step(hold)
        assert not info["success"].any()
        assert env.success_state(0).prop_pos["cube"][2] < 0.20

    # per-env independence: env 1 starts its lift reference where env 1 is
    with VecSimEnv(robot, 2, fps=50, seed=0, extra_xml=cube, success=Lifted("cube", 0.05)) as env:
        env.reset(seed=0)
        assert env._success[0] is not env._success[1]
        assert env._success[0] is not env.success_predicate


def test_final_success_follows_the_final_observation_convention(robot):
    """On an auto-reset the TERMINAL verdict goes to final_success; the
    index-aligned info["success"] describes the fresh state instead."""
    cube = CUBE_AT.format(x=0.3, y=0.0, z=0.5)
    # Terminate after exactly 3 control steps, by which point the cube has
    # fallen only ~2 cm: still above 0.4 m, so the terminal verdict is True.
    with VecSimEnv(
        robot,
        1,
        fps=50,
        seed=0,
        extra_xml=cube,
        success=Lifted("cube", 0.40, ref="absolute"),
        max_episode_steps=3,
    ) as env:
        obs = env.reset(seed=0)
        hold = obs["state"][:, : env.ndof].astype(np.float64)
        for step in range(1, 4):
            _o, _r, _te, tr, info = env.step(hold)
        assert tr[0] and info["reset_mask"][0] and step == 3
        assert info["final_success"] == [True]  # terminal state, pre-reset
        assert bool(info["success"][0]) is True  # fresh state, post-reset
        assert len(info["final_observation"]) == 1

        # After the reset the cube is back at its spawn: the next episode's
        # Lifted reference was re-captured there.
        assert env.success_state(0).prop_pos["cube"][2] == pytest.approx(0.5)


def test_image_observations_show_the_prop(robot):
    """A prop the policy must grasp has to be IN the images it is trained on:
    the render scene is built from the same extra_xml as the physics scene."""
    cube = CUBE_AT.format(x=0.25, y=0.0, z=0.25)
    with VecSimEnv(robot, 1, fps=50, seed=0, obs_images=True, image_size=(64, 64)) as plain:
        bare = plain.reset(seed=0)["image"][0].copy()
    with VecSimEnv(
        robot, 1, fps=50, seed=0, obs_images=True, image_size=(64, 64), extra_xml=cube
    ) as propped:
        obs = propped.reset(seed=0)
        # same joint layout in both models — that is what lets _obs hand the
        # scene the full qpos
        assert propped._scenes[0].model.nq == propped.model.nq
        assert obs["image"].shape == (1, 64, 64, 3)
        assert not np.array_equal(obs["image"][0], bare)  # the cube is there


def test_props_survive_domain_randomization(robot):
    """Model-level randomization recompiles each env's MJCF; the prop must come
    back with it, and the spawn offset must still land on the ARM's qpos."""
    from caliper_learn.randomize import RandomizationSpec

    spec = RandomizationSpec(
        mass_scale=(0.8, 1.2), joint_damping=(0.0, 0.1), spawn_jitter=(-0.05, 0.05)
    )
    cube = CUBE_AT.format(x=0.3, y=0.0, z=0.5)
    with VecSimEnv(
        robot, 2, fps=50, seed=0, extra_xml=cube, randomization=spec,
        success=Lifted("cube", 0.05),
    ) as env:
        obs = env.reset(seed=0)
        assert [m.nq for m in env._models] == [robot.ndof + 7] * 2  # prop kept
        assert obs["state"].shape == (2, 2 * robot.ndof)  # arm only, as always
        hold = obs["state"][:, : env.ndof].astype(np.float64)
        _o, _r, _te, _tr, info = env.step(hold)
        assert set(info) == {"reset_mask", "randomization", "success"}
        # the spawn offset moved the arm, not the cube
        assert env.success_state(0).prop_pos["cube"][:2] == pytest.approx((0.3, 0.0))


def test_predicate_naming_a_missing_prop_is_loud(robot):
    with VecSimEnv(robot, 1, fps=50, seed=0, success=Lifted("cube", 0.05)) as env:
        with pytest.raises(ValueError, match="no prop named 'cube'"):
            env.reset(seed=0)


# ----- eval + autopsy ---------------------------------------------------------------


@pytest.fixture(scope="module")
def reach_task_with_cube(robot):
    """The test_eval easy-reach task, plus a free cube parked out of the arm's
    way. The cube's fall is therefore identical in every episode — what varies
    per seed is WHEN the reach terminates, i.e. how far the cube has fallen by
    the time the episode is scored."""
    b = _bounds(robot)
    q_goal = b.mean(axis=1) + 0.3 * (0.5 * (b[:, 1] - b[:, 0]))
    n = robot.ndof

    def pos(q, f):
        p = robot.fk(q, f)
        return np.array([p[0][3], p[1][3], p[2][3]])

    frame = max(
        robot.frame_names(),
        key=lambda f: np.linalg.norm(pos([0.0] * n, f) - pos([0.4] * n, f)),
    )
    p = robot.fk([float(v) for v in q_goal], frame)
    task = reach_eval_task(
        robot, frame, [p[0][3], p[1][3], p[2][3]], tol=0.1, max_steps=80, fps=50,
        init_jitter=0.1,
    )
    task.extra_xml = CUBE_AT.format(x=5.0, y=5.0, z=1.0)
    return task, q_goal


class ProportionalPolicy:
    """Scripted baseline (test_eval's): drive a fraction of the gap each tick."""

    def __init__(self, q_goal, gain=0.6):
        self._q_goal = np.asarray(q_goal, dtype=np.float64)
        self._gain = float(gain)

    def __call__(self, state):
        q = state[: len(self._q_goal)].astype(np.float64)
        return q + self._gain * (self._q_goal - q)


def _cube_z_curve(robot, task, n):
    """Ground truth for the fixture: the cube's height after 1..n control
    steps, MEASURED on the same scene with the arm held still. The cube is
    parked far from the arm, so this curve is the same in every episode
    whatever the policy does."""
    with VecSimEnv(robot, 1, fps=task.fps, seed=0, extra_xml=task.extra_xml) as env:
        obs = env.reset(seed=0)
        hold = obs["state"][:, : env.ndof].astype(np.float64)
        zs = []
        for _ in range(n):
            env.step(hold)
            zs.append(env.success_state(0).prop_pos["cube"][2])
        return zs


def test_predicate_sourced_success_rate_is_hand_countable(robot, reach_task_with_cube):
    task, q_goal = reach_task_with_cube
    cfg = EvalConfig(n_episodes=6, base_seed=0)

    # Baseline: the same task scored the usual way (termination_fn fired).
    task.success_predicate = None
    base = evaluate(ProportionalPolicy(q_goal), task, cfg)
    assert base.success_criterion is None
    assert base.success_rate == 1.0  # every episode reaches the target
    steps = [e.steps for e in base.episodes]

    # Now score the SCENE instead. The zone's ceiling is placed BETWEEN the
    # cube's heights at steps k-1 and k, so the cube enters it at exactly
    # step k — no knife edge. An episode whose reach terminates before k is
    # judged on that terminal state, where the cube has not arrived yet:
    # terminating is no longer the same thing as succeeding.
    zs = _cube_z_curve(robot, task, task.max_steps)
    k = sorted(steps)[len(steps) // 2]  # mid-spread: some episodes each side
    ceiling = 0.5 * (zs[k - 1] + zs[k - 2])
    assert zs[k - 1] < ceiling < zs[k - 2]  # strictly decreasing, as assumed
    task.success_predicate = PlacedInZone(
        "cube", Zone(center=(5.0, 5.0, ceiling - 0.5), half=(0.2, 0.2, 0.5))
    )
    res = evaluate(ProportionalPolicy(q_goal), task, cfg)

    assert res.success_criterion == task.success_predicate.describe()
    # Hand count: succeed iff the episode would have run at least to step k.
    expected = [s >= k for s in steps]
    assert [e.success for e in res.episodes] == expected
    assert res.n_success == sum(expected)
    assert 0 < res.n_success < res.n_episodes  # the fixture really is mixed
    # and the episode ends at whichever came first, success or termination
    assert [e.steps for e in res.episodes] == [min(s, k) for s in steps]
    # Wilson is untouched: same counts in, same interval out.
    lo, hi = wilson_interval(res.n_success, res.n_episodes)
    assert (res.ci95_low, res.ci95_high) == (lo, hi)
    assert res.success_rate == res.n_success / res.n_episodes
    assert res.mean_steps_to_success == pytest.approx(
        sum(min(s, k) for s, ok in zip(steps, expected) if ok) / sum(expected)
    )
    assert [e.seed for e in res.episodes] == list(range(6))

    # the criterion reaches the human report and the machine one
    assert f"success: {res.success_criterion}" in render_text(res)
    import json

    assert json.loads(to_json(res))["success_criterion"] == res.success_criterion
    # and determinism still holds byte-for-byte
    assert to_json(evaluate(ProportionalPolicy(q_goal), task, cfg)) == to_json(res)


def test_unreachable_predicate_scores_zero_and_says_what_was_missed(
    robot, reach_task_with_cube
):
    task, q_goal = reach_task_with_cube
    task.success_predicate = Lifted("cube", 99.0, ref="absolute")  # never
    res = evaluate(ProportionalPolicy(q_goal), task, EvalConfig(n_episodes=3, base_seed=0))
    assert res.n_success == 0 and res.mean_steps_to_success is None
    assert [f.code for f in res.findings] == [ALL_EPISODES_FAILED]
    # E001 names the criterion that was missed, not "termination"
    assert res.success_criterion in res.findings[0].message
    task.success_predicate = None


def test_autopsy_verdict_states_the_success_criterion(robot, reach_task_with_cube):
    """The verdict may not quote a rate without saying what it is a rate of."""
    task, q_goal = reach_task_with_cube
    task.success_predicate = Lifted("cube", 0.5, ref="absolute")
    res = evaluate(ProportionalPolicy(q_goal), task, EvalConfig(n_episodes=2, base_seed=0))
    clean = {"findings": [], "total_episodes": 2, "total_frames": 2, "fps": 50}

    v = _verdict(clean, [], res, None)
    assert "closed-loop: " in v
    assert f"where success = {res.success_criterion}" in v

    # without a predicate the verdict is exactly what it always was
    task.success_predicate = None
    plain = evaluate(ProportionalPolicy(q_goal), task, EvalConfig(n_episodes=2, base_seed=0))
    assert "where success =" not in _verdict(clean, [], plain, None)


def test_absolute_ref_reset_does_not_require_the_prop():
    # Cross-face parity (found by review): Rust's SuccessTracker::reset collects
    # baselines only for initial-ref lifts, so a predicate over a prop missing
    # from the state must reset FINE and error at the first JUDGE — on both
    # faces. A reset-time lookup here made python fail one step earlier.
    p = Lifted("ghost", 0.3, ref="absolute")
    p.reset(_state(0.1))  # state only knows "cube" — must not raise
    with pytest.raises(ValueError, match="no prop named 'ghost'"):
        p(_state(0.1))
    # the initial form still captures (and still errors on a missing prop)
    q = Lifted("ghost", 0.3, ref="initial")
    with pytest.raises(ValueError, match="no prop named 'ghost'"):
        q.reset(_state(0.1))
