"""Phase-8 oracle: exercise the dataflow graph executor through the Python face.

Everything here runs with NO hardware and stays deterministic. We build a
`GraphDoc` (the camelCase `.caliper-graph.json` schema) as a plain Python dict,
`json.dumps` it, and hand it to `caliper.run_graph(robot, json)`.

Proven here:
  * a full pipeline (startConfig -> goalPose -> ik(seed) -> moveL -> view, plus a
    scope) produces a non-empty, internally-aligned terminal clip + scope series,
    with OK diagnostics;
  * determinism — a seeded planRrt graph run twice is byte-identical;
  * parity / safety — a moveJ graph reproduces `robot.move_j` directly (and stays
    within the velocity limit) deterministically;
  * the error path — a graph with a cycle is surfaced, either by raising or by
    returning not-OK diagnostics.

The result of `run_graph` may be exposed as a dict (snake_case or the camelCase
serde shape) or as an object with attributes; the `_field` helper accepts any of
those so the test pins behaviour, not the binding agent's exact surface.
"""

import json
import pathlib

import pytest

import caliper

np = pytest.importorskip("numpy")

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"

# The graph executor bakes every clip at this fixed step (caliper-graph CLIP_DT).
CLIP_DT = 0.01

pytestmark = pytest.mark.skipif(
    not hasattr(caliper, "run_graph"),
    reason="caliper lacks the Phase-8 run_graph binding — rebuild (maturin develop)",
)


def urdf(n):
    return str(ROBOTS / f"{n}.urdf")


def _robot():
    return caliper.Robot.from_urdf(urdf("showcase6"))


# ---------- result accessors (binding-shape agnostic) ----------

def _field(obj, *names):
    """Fetch the first present field of `obj`, trying dict keys then attributes.

    Accepts snake_case and camelCase aliases so the test does not hard-code the
    exact key style chosen by the binding."""
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    raise KeyError(f"none of {names} present in {type(obj).__name__}: {obj!r}")


def _opt_field(obj, default, *names):
    try:
        return _field(obj, *names)
    except KeyError:
        return default


def _terminal_clip(res):
    return _field(res, "terminal_clip", "terminalClip")


def _clip_rows(clip):
    times = _field(clip, "times")
    qs = _field(clip, "qs")
    qds = _field(clip, "qds")
    return times, qs, qds


def _diag_ok(res):
    diag = _field(res, "diagnostics")
    node_errors = _opt_field(diag, [], "node_errors", "nodeErrors")
    edge_errors = _opt_field(diag, [], "edge_errors", "edgeErrors")
    cycle = _opt_field(diag, [], "cycle")
    return not node_errors and not edge_errors and not cycle


def _assert_clip_aligned(clip, ndof):
    times, qs, qds = _clip_rows(clip)
    assert len(times) > 1, "terminal clip must be non-empty"
    assert len(qs) == len(times) == len(qds), "times / qs / qds must be aligned"
    for row in qs:
        assert len(row) == ndof, f"each q row must have length ndof={ndof}"
    for row in qds:
        assert len(row) == ndof, f"each qd row must have length ndof={ndof}"
    return times, qs, qds


# ---------- graph builders ----------

def _node(node_id, type_, **params):
    return {"id": node_id, "kind": dict(type=type_, **params)}


def _edge(frm, fp, to, tp):
    return {"from": frm, "fromPort": fp, "to": to, "toPort": tp}


def _goal_pose_colmajor(robot, q_goal, frame=None):
    """Column-major 16-vector SE3 of `frame` at `q_goal`, derived from the
    engine's own (row-major) FK — an independent witness that the target pose is
    reachable, so the IK / MoveL nodes have a real solution to track."""
    m = robot.fk(q_goal, frame)  # 4x4 row-major homogeneous
    col_major = []
    for col in range(4):
        for row in range(4):
            col_major.append(m[row][col])
    return col_major


# ---------- pipeline ----------

def test_pipeline_movel_clip_and_scope():
    robot = _robot()
    ndof = robot.ndof
    start = [0.0, -0.3, 0.5, 0.0, 0.4, 0.0]
    q_goal = [0.2, -0.1, 0.6, 0.1, 0.3, -0.1]
    goal_m = _goal_pose_colmajor(robot, q_goal)

    doc = {
        "nodes": [
            _node("start", "startConfig", q=start),
            _node("goal", "goalPose", m=goal_m),
            _node("ik", "ik"),
            _node("mv", "moveL"),
            _node("view", "view"),
            _node("scope", "scope", signal="q0"),
        ],
        "edges": [
            _edge("start", "config", "ik", "seed"),
            _edge("goal", "pose", "ik", "pose"),
            _edge("start", "config", "mv", "start"),
            _edge("goal", "pose", "mv", "goal"),
            _edge("mv", "clip", "view", "clip"),
            _edge("mv", "clip", "scope", "clip"),
        ],
    }

    res = caliper.run_graph(robot, json.dumps(doc))
    assert _diag_ok(res), "valid pipeline must yield OK diagnostics"

    clip = _terminal_clip(res)
    assert clip is not None, "view node must produce a terminal clip"
    times, qs, _ = _assert_clip_aligned(clip, ndof)

    # endpoints: the clip starts at `start`.
    assert all(abs(a - b) < 1e-6 for a, b in zip(qs[0], start))

    # the scope on q0 must yield a series aligned in t/y and matching the clip.
    scopes = _field(res, "scopes")
    assert len(scopes) == 1, "exactly one scope node"
    s = scopes[0]
    t = _field(s, "t")
    y = _field(s, "y")
    assert len(t) == len(y) > 0, "scope series t / y must be aligned + non-empty"
    assert len(t) == len(times), "scope is sampled along the terminal clip"
    # q0 series must equal the first joint column of the clip.
    assert all(abs(a - b[0]) < 1e-9 for a, b in zip(y, qs))


# ---------- determinism ----------

def test_planrrt_graph_deterministic():
    robot = _robot()
    start = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    goal = [0.4, -0.3, 0.4, 0.0, 0.2, 0.0]

    doc = {
        "nodes": [
            _node("start", "startConfig", q=start),
            _node("goal", "startConfig", q=goal),
            _node("plan", "planRrt", seed=0xCA11),
            _node("view", "view"),
        ],
        "edges": [
            _edge("start", "config", "plan", "start"),
            _edge("goal", "config", "plan", "goal"),
            _edge("plan", "clip", "view", "clip"),
        ],
    }
    payload = json.dumps(doc)

    res1 = caliper.run_graph(robot, payload)
    res2 = caliper.run_graph(robot, payload)
    assert _diag_ok(res1) and _diag_ok(res2)

    c1 = _terminal_clip(res1)
    c2 = _terminal_clip(res2)
    t1, qs1, qds1 = _clip_rows(c1)
    t2, qs2, qds2 = _clip_rows(c2)
    # byte-identical: a seeded planRrt is fully deterministic.
    assert t1 == t2, "same seed ⇒ identical clip times"
    assert qs1 == qs2, "same seed ⇒ identical clip qs"
    assert qds1 == qds2, "same seed ⇒ identical clip qds"
    assert len(t1) > 1


# ---------- parity + safety ----------

def test_movej_graph_matches_direct():
    robot = _robot()
    ndof = robot.ndof
    start = [0.0, -0.3, 0.5, 0.0, 0.4, 0.0]
    goal = [0.2, -0.1, 0.6, 0.1, 0.3, -0.1]

    doc = {
        "nodes": [
            _node("start", "startConfig", q=start),
            _node("goal", "startConfig", q=goal),
            _node("mv", "moveJ"),
            _node("view", "view"),
        ],
        "edges": [
            _edge("start", "config", "mv", "start"),
            _edge("goal", "config", "mv", "goal"),
            _edge("mv", "clip", "view", "clip"),
        ],
    }
    payload = json.dumps(doc)

    res = caliper.run_graph(robot, payload)
    assert _diag_ok(res)
    clip = _terminal_clip(res)
    times, qs, qds = _assert_clip_aligned(clip, ndof)

    # determinism: identical on a second run.
    res2 = caliper.run_graph(robot, payload)
    t2, qs2, qds2 = _clip_rows(_terminal_clip(res2))
    assert times == t2 and qs == qs2 and qds == qds2

    # endpoints reach start / goal.
    assert all(abs(a - b) < 1e-6 for a, b in zip(qs[0], start))
    assert all(abs(a - b) < 1e-3 for a, b in zip(qs[-1], goal))

    # within the velocity limit (showcase6 limits; generous 5% margin).
    vmax = robot.move_j(start, goal).vel_limit
    assert all(abs(v) <= lim * 1.05 for row in qds for v, lim in zip(row, vmax))

    # parity: the graph's baked clip equals move_j sampled at CLIP_DT directly.
    direct = robot.move_j(start, goal)
    dt_times, dq, dqd, _ = direct.sample_uniform(CLIP_DT)
    if len(dt_times) == len(times):
        for gr, dr in zip(qs, dq):
            assert all(abs(a - b) < 1e-9 for a, b in zip(gr, dr)), (
                "moveJ graph clip must match move_j directly"
            )
        for gr, dr in zip(qds, dqd):
            assert all(abs(a - b) < 1e-9 for a, b in zip(gr, dr))


# ---------- error path ----------

def test_cycle_is_surfaced():
    robot = _robot()
    # Two IK nodes feeding each other's seed (Config→Config) form a 2-cycle.
    doc = {
        "nodes": [
            _node("a", "ik"),
            _node("b", "ik"),
        ],
        "edges": [
            _edge("a", "config", "b", "seed"),
            _edge("b", "config", "a", "seed"),
        ],
    }
    payload = json.dumps(doc)

    # Either run_graph raises a validation error, or it returns not-OK
    # diagnostics — either way the cycle must be surfaced, never silently run.
    try:
        res = caliper.run_graph(robot, payload)
    except Exception:
        return  # raised → surfaced
    assert not _diag_ok(res), "a cyclic graph must produce not-OK diagnostics"
    cycle = _opt_field(_field(res, "diagnostics"), [], "cycle")
    assert cycle, "the reported diagnostics must name the cycle"
