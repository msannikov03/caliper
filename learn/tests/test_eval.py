"""Seeded eval harness tests: a scripted controller clears an easy reach task,
a null policy scores ~0, determinism is byte-exact on the serialized result,
sweep ranks good above null, Wilson CI matches hand arithmetic, and every
finding has a positive (crafted defect detected) AND a negative (clean run,
zero findings) case. CPU-only, seconds."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
caliper = pytest.importorskip("caliper")

from caliper_learn.collect import _bounds, _resolve_urdf  # noqa: E402
from caliper_learn.eval import (  # noqa: E402
    ALL_EPISODES_FAILED,
    SEED_LOTTERY,
    ZERO_REWARD_SIGNAL,
    EvalConfig,
    EvalResult,
    EvalTask,
    evaluate,
    reach_eval_task,
    render_text,
    sweep,
    to_json,
    wilson_interval,
)


@pytest.fixture(scope="module")
def robot():
    # collide_arm: the standard planner fixture (has inertials -> MJCF-exportable)
    return caliper.Robot.from_urdf(_resolve_urdf("planner", None))


@pytest.fixture(scope="module")
def easy_reach(robot):
    """Reach the FK point of q_goal = mid + 0.3*half — well separated from both
    the jittered starts (around the limit midpoints) and the null policy's q=0
    attractor (~0.76 m away for this fixture), so success discriminates."""
    b = _bounds(robot)
    q_goal = b.mean(axis=1) + 0.3 * (0.5 * (b[:, 1] - b[:, 0]))
    frame = _moving_frame(robot)
    pose = robot.fk([float(v) for v in q_goal], frame)
    target = [pose[0][3], pose[1][3], pose[2][3]]
    task = reach_eval_task(
        robot, frame, target, tol=0.1, max_steps=80, fps=50, init_jitter=0.1
    )
    return task, q_goal


def _moving_frame(robot) -> str:
    """Pick the frame whose world position moves most between two configs, so
    the reach reward actually depends on q."""
    n = robot.ndof
    qa, qb = [0.0] * n, [0.4] * n

    def pos(q, f):
        p = robot.fk(q, f)
        return np.array([p[0][3], p[1][3], p[2][3]])

    return max(robot.frame_names(), key=lambda f: np.linalg.norm(pos(qa, f) - pos(qb, f)))


class ProportionalPolicy:
    """Scripted baseline: drive a fraction of the gap toward q_goal each tick.
    Deterministic, stateless — the 'known good' reference in the ranking."""

    def __init__(self, q_goal, gain=0.6):
        self._q_goal = np.asarray(q_goal, dtype=np.float64)
        self._gain = float(gain)

    def __call__(self, state):
        q = state[: len(self._q_goal)].astype(np.float64)
        return q + self._gain * (self._q_goal - q)


def null_policy(state):
    """Zero-action baseline: always commands the q=0 target."""
    return np.zeros(len(state) // 2)


class FlakyPolicy:
    """Crafted defect: proportional on odd episodes, null on even — success
    flips seed-to-seed, the seed-lottery smell."""

    def __init__(self, q_goal):
        self._good = ProportionalPolicy(q_goal)
        self._episode = -1

    def reset(self):
        self._episode += 1

    def __call__(self, state):
        return self._good(state) if self._episode % 2 == 1 else null_policy(state)


# ----- evaluate: scripted baselines --------------------------------------------


def test_proportional_policy_clears_easy_reach(easy_reach):
    task, q_goal = easy_reach
    res = evaluate(ProportionalPolicy(q_goal), task, EvalConfig(n_episodes=8, base_seed=0))
    assert res.success_rate > 0.8
    assert res.n_success == sum(e.success for e in res.episodes)
    assert res.mean_steps_to_success is not None and res.mean_steps_to_success < task.max_steps
    # successful episodes ended within tolerance of the target
    for e in res.episodes:
        if e.success:
            assert e.final_distance is not None and e.final_distance < 0.1
    # per-episode seeds are always present and reproduce the config's stream
    assert [e.seed for e in res.episodes] == list(range(8))
    # NEGATIVE finding case: a clean, competent run raises nothing
    assert res.findings == ()


def test_null_policy_scores_zero_and_is_flagged(easy_reach):
    task, _q_goal = easy_reach
    res = evaluate(null_policy, task, EvalConfig(n_episodes=4, base_seed=0))
    assert res.success_rate == 0.0
    assert res.mean_steps_to_success is None
    assert all(e.steps == task.max_steps for e in res.episodes)
    # never got close: parked ~0.76 m from the target for this fixture
    assert all(e.final_distance > 0.5 for e in res.episodes)
    # POSITIVE finding case: the all-episodes-failed smell, and only that one
    assert [f.code for f in res.findings] == [ALL_EPISODES_FAILED]
    f = res.findings[0]
    assert f.severity == "warn" and f.fix_hint  # plain-English + actionable


# ----- determinism ---------------------------------------------------------------


def test_determinism_byte_identical_json(easy_reach):
    task, q_goal = easy_reach
    cfg = EvalConfig(n_episodes=3, base_seed=11)
    a = evaluate(ProportionalPolicy(q_goal), task, cfg)
    b = evaluate(ProportionalPolicy(q_goal), task, cfg)
    assert to_json(a) == to_json(b)  # exact bytes, not approx
    assert a == b  # and the dataclasses themselves
    # a different base seed is a genuinely different run
    c = evaluate(ProportionalPolicy(q_goal), task, EvalConfig(n_episodes=3, base_seed=12))
    assert to_json(a) != to_json(c)


# ----- findings: remaining positive/negative pairs -------------------------------


def test_seed_lottery_finding_fires_on_flaky_policy(easy_reach):
    task, q_goal = easy_reach
    res = evaluate(FlakyPolicy(q_goal), task, EvalConfig(n_episodes=4, base_seed=0))
    assert res.n_success == 2  # crafted: succeeds on episodes 1 and 3
    codes = [f.code for f in res.findings]
    assert SEED_LOTTERY in codes and ALL_EPISODES_FAILED not in codes
    # negative case: the same mixed rate with a TIGHT interval stays silent —
    # covered structurally by test_proportional (0 findings at rate 1.0) plus
    # the width>0.5 gate spot-checked here:
    lo, hi = wilson_interval(50, 100)
    assert (hi - lo) < 0.5  # at n=100 the same 50% rate would NOT be a lottery


def test_zero_reward_finding_fires_without_reward_fn(easy_reach, robot):
    task, q_goal = easy_reach
    silent = EvalTask(
        robot=robot,
        reward_fn=None,  # crafted defect: no reward wired
        termination_fn=task.termination_fn,
        max_steps=task.max_steps,
        fps=task.fps,
        init_jitter=task.init_jitter,
    )
    res = evaluate(ProportionalPolicy(q_goal), silent, EvalConfig(n_episodes=3, base_seed=0))
    assert res.success_rate == 1.0  # still succeeds — but returns say nothing
    assert res.mean_return == 0.0 and res.median_return == 0.0
    assert [f.code for f in res.findings] == [ZERO_REWARD_SIGNAL]
    # negative case: the rewarded runs above (test_proportional/null) carry
    # nonzero returns and never raise E003
    rewarded = evaluate(null_policy, task, EvalConfig(n_episodes=2, base_seed=0))
    assert ZERO_REWARD_SIGNAL not in [f.code for f in rewarded.findings]


# ----- sweep ----------------------------------------------------------------------


def test_sweep_ranks_good_above_null(easy_reach):
    task, q_goal = easy_reach
    cfg = EvalConfig(n_episodes=4, base_seed=0)
    ranked = sweep({"good": ProportionalPolicy(q_goal), "null": null_policy}, task, cfg)
    assert [e.name for e in ranked] == ["good", "null"]
    assert ranked[0].result.success_rate > ranked[1].result.success_rate
    assert all(e.path is None for e in ranked)  # in-memory candidates
    # each row is a full EvalResult with its seeds — any episode reproducible
    for e in ranked:
        assert isinstance(e.result, EvalResult)
        assert [ep.seed for ep in e.result.episodes] == list(range(4))
    with pytest.raises(ValueError):
        sweep({}, task, cfg)


# ----- Wilson interval -------------------------------------------------------------


def test_wilson_interval_matches_hand_computation():
    # Hand-computed with z = 1.959963984540054, p̂ = 8/10:
    #   denom  = 1 + z²/10                  = 1.3841458820694124
    #   center = (0.8 + z²/20) / denom      = 0.7167401622...
    #   half   = z·sqrt(0.8·0.2/10 + z²/400) / denom
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.49016247153664183, rel=1e-12)
    assert hi == pytest.approx(0.9433178485456247, rel=1e-12)
    # edge rates stay honest (non-degenerate) and clipped to [0, 1]
    lo0, hi0 = wilson_interval(0, 5)
    assert lo0 == 0.0 and hi0 == pytest.approx(0.43448246478317476, rel=1e-12)
    lo5, hi5 = wilson_interval(5, 5)
    assert lo5 == pytest.approx(0.5655175352168251, rel=1e-12) and hi5 == 1.0
    # containment + monotone shrink with n
    for s, n in [(3, 7), (0, 3), (9, 9)]:
        lo, hi = wilson_interval(s, n)
        assert 0.0 <= lo <= s / n <= hi <= 1.0
    lo20, hi20 = wilson_interval(10, 20)
    lo200, hi200 = wilson_interval(100, 200)
    assert (hi200 - lo200) < (hi20 - lo20)
    with pytest.raises(ValueError):
        wilson_interval(3, 0)
    with pytest.raises(ValueError):
        wilson_interval(6, 5)


# ----- reporting -------------------------------------------------------------------


def test_render_text_and_json_shapes(easy_reach):
    task, q_goal = easy_reach
    cfg = EvalConfig(n_episodes=2, base_seed=5)
    res = evaluate(ProportionalPolicy(q_goal), task, cfg)
    txt = render_text(res)
    # seeds ALWAYS in the output — reproduce any single episode
    assert "5" in txt and "6" in txt and "wilson95" in txt
    ranked = sweep({"good": ProportionalPolicy(q_goal)}, task, cfg)
    stxt = render_text(ranked)
    assert "good" in stxt and "rank" in stxt

    import json

    payload = json.loads(to_json(res))
    assert [e["seed"] for e in payload["episodes"]] == [5, 6]
    spayload = json.loads(to_json(ranked))
    assert spayload[0]["name"] == "good"
    assert [e["seed"] for e in spayload[0]["result"]["episodes"]] == [5, 6]
    with pytest.raises(TypeError):
        to_json({"not": "a result"})
    with pytest.raises(TypeError):
        render_text(42)


def test_unsupported_policy_type_raises(easy_reach):
    task, _ = easy_reach
    with pytest.raises(TypeError):
        evaluate(42, task, EvalConfig(n_episodes=1))
    with pytest.raises(ValueError):
        evaluate(lambda s: np.zeros(1), task, EvalConfig(n_episodes=1))  # wrong shape
    with pytest.raises(ValueError):
        evaluate(null_policy, task, EvalConfig(n_episodes=0))
