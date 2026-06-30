"""Edge-case STRESS oracle: narrow-passage planning, near-pi IK, at-joint-limit IK.

These exercise the failure boundary of the planner and IK through the Python face
with NO hardware. The guiding contract everywhere is HONESTY: the engine may
solve a hard instance OR report failure, but it must never return a *wrong*
answer dressed up as success.

  (1) narrow passage -- two world boxes leave a thin gap across the arm's swept
      arc. The planner must EITHER return a path that is genuinely collision-free
      (re-checked edge-by-edge by an INDEPENDENT CollisionModel, not just the
      planner's own verify) OR honestly raise (no path within the iteration
      budget). We also prove verify() does not lie: a path through a config we
      *independently* know collides must be rejected.

  (2) near-pi IK -- a target whose orientation differs from the seed by ~pi.
      A single damped-Newton pass from the identity seed can stall at the
      antipodal orientation (the rotation-error vector is sign-ambiguous at
      exactly pi); the solver's multi-restart is what rescues it. We assert the
      result is honest: success ⇒ FK(q) reproduces the target; otherwise
      success is False (never a silent near-miss). No NaNs either way.

  (3) at-limit IK -- a target that is the FK image of a config sitting EXACTLY on
      a joint's position limit. The returned q must respect every limit (the
      solver clamps each step) and the solver must terminate (no NaN / no hang).

All scene/precondition facts are queried LIVE from the engine's own
CollisionModel / FK inside each test (not hard-coded), so the tests self-validate
against the actual geometry rather than against my arithmetic. Deterministic
seeds; small fixtures; fast."""

import math
import pathlib

import pytest

import caliper

np = pytest.importorskip("numpy")

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"


def urdf(n):
    return str(ROBOTS / f"{n}.urdf")


# Phase-6 planning bindings (Planner/CollisionModel) gate the planning tests.
_HAVE_PLANNING = all(hasattr(caliper, c) for c in ("Planner", "CollisionModel"))
# Phase-1 IK binding gates the IK tests.
_HAVE_IK = hasattr(caliper.Robot, "ik")

needs_planning = pytest.mark.skipif(
    not _HAVE_PLANNING,
    reason="caliper lacks Planner/CollisionModel — rebuild (maturin develop)",
)
needs_ik = pytest.mark.skipif(
    not _HAVE_IK, reason="caliper.Robot lacks ik — rebuild (maturin develop)"
)

SEED = 0xCA11


def _arm():
    return caliper.Robot.from_urdf(urdf("collide_arm"))


# A two-box "wall with a slot" placed across the arm's swept arc. Each entry is
# (center, half_extents). The boxes sit at x≈[0.30, 0.55] (so the straight-up
# start at x≈0 is clear) and leave a thin vertical gap at z≈[0.56, 0.72]. The
# folded-down goal stays below z≈0.42, clearing the lower box. See the module
# docstring; all of this is re-checked live below rather than trusted.
def _slot_boxes():
    return [
        ((0.425, 0.0, 0.49), (0.125, 0.30, 0.07)),  # lower wall:  z in [0.42, 0.56]
        ((0.425, 0.0, 0.81), (0.125, 0.30, 0.09)),  # upper wall:  z in [0.72, 0.90]
    ]


def _lerp(a, b, t):
    return [ai + (bi - ai) * t for ai, bi in zip(a, b)]


def _edge_free(cm, a, b, substeps=24):
    """True iff every interpolated config on edge a→b is collision-free, queried
    through a *standalone* CollisionModel (independent of the planner)."""
    for k in range(substeps + 1):
        if cm.query(_lerp(a, b, k / substeps))["collision"]:
            return False
    return True


# ===================== (1) narrow-passage planning =====================


@needs_planning
def test_narrow_passage_plan_is_honest_and_verify_does_not_lie():
    start, goal = [0.0, 0.0, 0.0], [1.2, 0.0, 0.0]
    boxes = _slot_boxes()

    # Independent collision oracle for the same scene.
    cm = caliper.CollisionModel(_arm(), boxes=boxes)

    # --- precondition: endpoints must be collision-free (else the scene, not the
    #     planner, is what we'd be testing) ---
    assert not cm.query(start)["collision"], "start config must be collision-free"
    assert not cm.query(goal)["collision"], "goal config must be collision-free"

    # --- non-vacuity: the naive straight-line joint path must actually hit the
    #     obstacle, AND we capture a concrete colliding witness config to drive
    #     the verify()-honesty check below. If nothing collides, the scene is
    #     trivial and the whole test would prove nothing. ---
    bad = None
    for k in range(1, 200):
        q = _lerp(start, goal, k / 200)
        if cm.query(q)["collision"]:
            bad = q
            break
    assert bad is not None, (
        "straight-line path never collides — the slot scene is not actually "
        "blocking the arc, so this test is vacuous"
    )

    # --- verify() must NOT report a colliding path as collision-free. The path
    #     start→bad→goal contains a config we *independently* know collides. ---
    colliding_path = [start, bad, goal]
    assert not _edge_free(cm, start, bad) or not _edge_free(cm, bad, goal), (
        "independent CollisionModel sanity: the witness path must collide"
    )
    p = caliper.Planner(_arm(), boxes=boxes, seed=SEED)
    assert p.verify(colliding_path) is False, (
        "verify() falsely reported a colliding path as collision-free"
    )

    # --- the planner is HONEST: either a genuinely collision-free path, or a
    #     raised failure within the iteration budget. ---
    try:
        path = p.plan(start, goal)
    except ValueError as e:
        # Honest failure. It must be an "unreachable within budget" verdict, NOT
        # a start/goal-in-collision claim (we proved those are free above).
        msg = str(e).lower()
        assert "collision-free path found" in msg or "iterations" in msg, (
            f"unexpected planner failure for a free-endpoint instance: {e}"
        )
        return

    # Success path: re-verify every edge through the planner AND, independently,
    # through the standalone CollisionModel — a bug shared by plan()/verify()
    # cannot pass the second check.
    assert path[0] == start and path[-1] == goal, "path must connect the endpoints"
    assert p.verify(path), "planner returned a path its own verify() rejects"
    assert all(
        _edge_free(cm, a, b) for a, b in zip(path[:-1], path[1:])
    ), "planner's 'collision-free' path collides under an independent re-check"


@needs_planning
def test_narrow_passage_deterministic():
    start, goal = [0.0, 0.0, 0.0], [1.2, 0.0, 0.0]
    boxes = _slot_boxes()

    def run():
        p = caliper.Planner(_arm(), boxes=boxes, seed=SEED)
        try:
            return ("ok", p.plan(start, goal))
        except ValueError as e:
            return ("fail", str(e))

    a, b = run(), run()
    assert a == b, "same seed ⇒ identical outcome (path or failure), got a≠b"


# ===================== (2) near-pi IK =====================


def _to_colmajor(m):
    """caliper.fk returns a row-major 4x4 (list of rows). Robot.ik wants the
    target as 4 COLUMNS each of length 4 (column-major). Transpose."""
    return [[m[r][c] for r in range(4)] for c in range(4)]


def _fk_pos(robot, q):
    m = robot.fk(q)
    return (m[0][3], m[1][3], m[2][3])


@needs_ik
def test_near_pi_ik_is_honest():
    """Target requires a ~pi reorientation from the seed. collide_arm's limits
    ([-3.3, 3.3]) admit it. Large-rotation behaviour: a lone Newton step from the
    identity seed can stall at the antipodal orientation; the solver's restarts
    are what let it converge. We require honesty, not necessarily success."""
    robot = _arm()
    seed = [0.0, 0.0, 0.0]
    # q* puts ~3.1 rad of cumulative rotation on the (planar, about-y) chain — an
    # end-effector orientation ~pi away from the seed's. The target is the engine's
    # OWN FK image of q*, so it is exactly reachable in principle.
    q_star = [3.0, 0.2, -0.1]
    target = _to_colmajor(robot.fk(q_star))

    res = robot.ik(target, seed)
    q = res["q"]

    # No silent NaN / non-termination.
    assert len(q) == robot.ndof
    assert all(math.isfinite(v) for v in q), f"IK returned non-finite q: {q}"
    assert math.isfinite(res["residual"])

    if res["success"]:
        # success ⇒ FK(q) actually reproduces the target pose (no silent miss).
        got, want = _fk_pos(robot, q), _fk_pos(robot, q_star)
        assert all(abs(a - b) < 1e-6 for a, b in zip(got, want)), (
            f"IK reported success but FK position {got} != target {want}"
        )
        # residual must agree with the success verdict (tol_pos/tol_rot = 1e-9).
        assert res["residual"] < 1e-6, (
            f"success with residual {res['residual']:.3e} — verdict inconsistent"
        )
    else:
        # Honest failure: the residual must be genuinely non-trivial (not a near
        # hit relabeled as failure). A real stall sits orders above tol (1e-9).
        assert res["residual"] > 1e-7, (
            f"reported failure but residual {res['residual']:.3e} is ~converged"
        )


@needs_ik
def test_pi_orientation_ik_no_silent_wrong_answer():
    """A clean half-turn: target = FK at q=[pi, 0, 0] (orientation exactly pi from
    the seed, the worst case for the rotation-error sign ambiguity). Whatever the
    verdict, it must be honest."""
    robot = _arm()
    q_star = [math.pi, 0.0, 0.0]
    target = _to_colmajor(robot.fk(q_star))
    res = robot.ik(target, [0.0, 0.0, 0.0])
    q = res["q"]
    assert all(math.isfinite(v) for v in q), f"non-finite q at pi target: {q}"
    if res["success"]:
        got, want = _fk_pos(robot, q), _fk_pos(robot, q_star)
        assert all(abs(a - b) < 1e-6 for a, b in zip(got, want)), (
            f"success but FK {got} != target {want} (silent wrong answer)"
        )


# ===================== (3) at-joint-limit IK =====================


@needs_ik
def test_at_joint_limit_ik_respects_limits_and_terminates():
    """limit_arm's j2 range is [-0.5, 0.5]. The target is the FK image of
    q=[1.0, 0.5], i.e. j2 sits EXACTLY on its upper limit. The solver must return
    a q that respects every limit (it clamps each step) and must not NaN / hang."""
    robot = caliper.Robot.from_urdf(urdf("limit_arm"))
    limits = robot.joint_limits  # list of (lo, hi) | None
    assert robot.ndof == 2

    q_star = [1.0, 0.5]  # j2 exactly at its upper limit (0.5)
    target = _to_colmajor(robot.fk(q_star))

    res = robot.ik(target, [0.0, 0.0])
    q = res["q"]

    # terminates with a real answer (no NaN / inf — a hang would never get here).
    assert len(q) == 2
    assert all(math.isfinite(v) for v in q), f"IK returned non-finite q: {q}"

    # every coordinate respects its position limit (within a tight tolerance —
    # the engine clamps each Newton step to [lo, hi]).
    TOL = 1e-9
    for i, lim in enumerate(limits):
        if lim is None:
            continue
        lo, hi = lim
        assert lo - TOL <= q[i] <= hi + TOL, (
            f"joint {i} = {q[i]} violates limit [{lo}, {hi}]"
        )

    if res["success"]:
        # The target is exactly reachable only with j2 on its boundary, so a
        # successful solve must land j2 at the upper limit and reproduce the pose.
        assert abs(q[1] - 0.5) < 1e-6, (
            f"j2 should sit on its 0.5 limit, got {q[1]}"
        )
        got, want = _fk_pos(robot, q), _fk_pos(robot, q_star)
        assert all(abs(a - b) < 1e-6 for a, b in zip(got, want)), (
            f"success but FK {got} != target {want}"
        )
