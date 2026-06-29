"""Phase-6 oracle: exercise the planner + reachability through the Python face.

Core guarantees proven here with NO hardware: determinism (same seed ⇒ identical
path), the collision-free guarantee (the path's own `verify` re-checks every edge
at finer resolution, AND an independent standalone-CollisionModel re-check so a
bug shared by plan()/verify() can't pass), connectivity (endpoints), retiming
endpoints, and the three
reachability verdicts — with the "reachable"/"blocked" target pose derived from
PINOCCHIO forward kinematics (an independent cross-check of the target itself)."""

import math
import pathlib

import pytest

import caliper

np = pytest.importorskip("numpy")

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"


def urdf(n):
    return str(ROBOTS / f"{n}.urdf")


pytestmark = pytest.mark.skipif(
    not all(hasattr(caliper, c) for c in ("Planner", "ReachChecker")),
    reason="caliper lacks Phase-6 planning bindings — rebuild (maturin develop)",
)


def _arm():
    return caliper.Robot.from_urdf(urdf("collide_arm"))


def _scene_box():
    return [((0.6, 0.0, 0.3), (0.15, 0.15, 0.15))]


def _recheck_collision_free(path, ground, boxes, substeps=16):
    """INDEPENDENT collision re-check of a planned path.

    `Planner.verify` re-checks the path with the planner's OWN checker, so a
    bug shared between plan() and verify() would pass both (self-consistent, not
    external). Here we densely interpolate every edge and query a *separately
    constructed* `CollisionModel` over the same scene — a check that does not go
    through the planner at all. Returns True iff every sampled config is free."""
    cm = caliper.CollisionModel(_arm(), ground=ground, boxes=boxes)
    for a, b in zip(path[:-1], path[1:]):
        for k in range(substeps + 1):
            t = k / substeps
            q = [ai + (bi - ai) * t for ai, bi in zip(a, b)]
            if cm.query(q)["collision"]:
                return False
    return True


# ---------- planning ----------

def test_plan_deterministic_and_collision_free():
    start, goal = [0.0, 0.0, 0.0], [0.4, -0.4, 0.4]
    p1 = caliper.Planner(_arm(), ground=-0.1, boxes=_scene_box(), seed=0xCA11)
    p2 = caliper.Planner(_arm(), ground=-0.1, boxes=_scene_box(), seed=0xCA11)
    path1 = p1.plan(start, goal)
    path2 = p2.plan(start, goal)
    assert path1 == path2, "same seed ⇒ identical path"
    assert p1.verify(path1), "every edge must be collision-free (re-verified)"
    # Independent witness: re-check via a standalone CollisionModel, NOT the
    # planner's own verify (see _recheck_collision_free).
    assert _recheck_collision_free(path1, ground=-0.1, boxes=_scene_box()), (
        "path must be collision-free under an independent CollisionModel re-check"
    )
    assert path1[0] == start and path1[-1] == goal


def test_plan_to_pose_via_pinocchio_target():
    pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")
    q = [0.3, -0.3, 0.3]
    target = _pin_fk_target(pin, "collide_arm", q, "l3")
    p = caliper.Planner(_arm(), seed=0xCA11)
    path = p.plan_to_pose([0.0, 0.0, 0.0], target)
    assert len(path) >= 2
    assert p.verify(path), "planned path to the pose must be collision-free"


def test_plan_trajectory_endpoints():
    start, goal = [0.0, 0.0, 0.0], [0.4, -0.4, 0.4]
    p = caliper.Planner(_arm(), seed=0xCA11)
    ts, qs, qds = p.plan_trajectory(start, goal, dt=0.02)
    assert len(ts) == len(qs) == len(qds) >= 2
    assert all(abs(a - b) < 1e-6 for a, b in zip(qs[0], start))
    assert all(abs(a - b) < 1e-3 for a, b in zip(qs[-1], goal))
    # velocities within the model limit (collide_arm vel=5 rad/s; generous margin)
    assert all(abs(v) <= 5.0 * 1.05 for row in qds for v in row)


# ---------- reachability ----------

def test_reach_unreachable_far():
    rc = caliper.ReachChecker(_arm())
    v = rc.status([1, 0, 0, 0, 1, 0, 0, 0, 1, 10.0, 0.0, 0.0])
    assert v["status"] == "unreachable"


def test_reach_reachable_then_blocked():
    pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")
    q = [0.3, -0.3, 0.3]
    target = _pin_fk_target(pin, "collide_arm", q, "l3")
    # free space → reachable
    assert caliper.ReachChecker(_arm()).status(target)["status"] == "reachable"
    # a big box enveloping that pose → every IK solution collides → blocked
    c = target[9:12]
    boxes = [((c[0], c[1], c[2]), (1.0, 1.0, 1.0))]
    v = caliper.ReachChecker(_arm(), boxes=boxes).status(target)
    assert v["status"] == "blocked", v


def _pin_fk_target(pin, name, q, frame):
    """Cartesian pose (9 row-major R, then tx,ty,tz) of `frame` at config `q`,
    computed by Pinocchio — an independent witness that the pose is the FK image."""
    model = pin.buildModelFromUrdf(urdf(name))
    data = model.createData()
    pin.forwardKinematics(model, data, np.array(q, dtype=float))
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId(frame)
    m = data.oMf[fid]
    r = m.rotation
    t = m.translation
    return [
        r[0, 0], r[0, 1], r[0, 2],
        r[1, 0], r[1, 1], r[1, 2],
        r[2, 0], r[2, 1], r[2, 2],
        float(t[0]), float(t[1]), float(t[2]),
    ]
