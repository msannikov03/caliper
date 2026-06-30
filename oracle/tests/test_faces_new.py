"""Cross-validate the Wave-5 face bindings (analytic IK, RRT* plan_optimal,
joint-offset calibration) end-to-end through the Python face."""

import numpy as np
import pytest

caliper = pytest.importorskip("caliper")

FIX = "oracle/fixtures/robots"


def _robot(name):
    return caliper.Robot.from_urdf(f"{FIX}/{name}.urdf")


# ---------- analytic IK ----------
def test_analytic_ik_branches_reproduce_target_and_match_numeric():
    r = _robot("showcase6")
    rng = np.random.default_rng(0)
    compared = 0
    for _ in range(60):
        q_true = rng.uniform(-1.0, 1.0, r.ndof).tolist()
        T = np.array(r.fk(q_true))  # row-major
        seed = (np.array(q_true) + rng.uniform(-0.2, 0.2, r.ndof)).tolist()
        branches = r.analytic_ik(T.T.tolist(), seed)  # col-major target
        assert branches is not None and len(branches) >= 1
        # every branch reproduces the target tip
        for b in branches:
            err = np.linalg.norm(np.array(r.fk(b))[:3, 3] - T[:3, 3])
            assert err < 1e-7, f"analytic branch tip off by {err:.2e}"
        # seed-nearest (first) agrees with the numeric solver when it converges
        num = r.ik(T.T.tolist(), seed)
        if num["success"]:
            tip_a = np.array(r.fk(branches[0]))[:3, 3]
            tip_n = np.array(r.fk(num["q"]))[:3, 3]
            assert np.linalg.norm(tip_a - tip_n) < 1e-6
            compared += 1
    assert compared >= 40, f"numeric IK converged on only {compared}/60"


def test_analytic_ik_none_on_non_spherical_wrist():
    for name in ("toy", "redundant7", "prismatic"):
        r = _robot(name)
        T = np.array(r.fk([0.1] * r.ndof))
        assert r.analytic_ik(T.T.tolist(), [0.0] * r.ndof) is None


# ---------- RRT* plan_optimal ----------
def _free(cm, q):
    return not cm.query(list(q))["collision"]


def test_plan_optimal_is_collision_free_and_not_worse_than_rrt_connect():
    r = _robot("collide_arm")
    ground, boxes = -0.1, [((0.6, 0.0, 0.3), (0.15, 0.15, 0.15))]
    cm = caliper.CollisionModel(r, ground=ground, boxes=boxes)
    P = caliper.Planner(r, ground=ground, boxes=boxes, seed=7)
    rng = np.random.default_rng(3)

    def sample_free():
        for _ in range(500):
            q = [float(x) for x in rng.uniform(-1.0, 1.0, r.ndof)]
            if _free(cm, q):
                return q
        pytest.skip("could not sample a free config")

    start, goal = sample_free(), sample_free()

    def cost(path):
        return sum(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])) for i in range(len(path) - 1))

    opt = P.plan_optimal(start, goal, 4000)
    rrt = P.plan(start, goal)
    # endpoints exact
    assert np.allclose(opt[0], start) and np.allclose(opt[-1], goal)
    # independently collision-free along the path (fine resampling)
    for i in range(len(opt) - 1):
        a, b = np.array(opt[i]), np.array(opt[i + 1])
        for t in np.linspace(0, 1, 25):
            assert _free(cm, (a + t * (b - a)).tolist())
    # RRT* should not be meaningfully worse than RRT-Connect (both smoothed)
    assert cost(opt) <= cost(rrt) * 1.10 + 1e-6, f"opt {cost(opt):.3f} vs rrt {cost(rrt):.3f}"


# ---------- joint-offset calibration ----------
def test_calibrate_recovers_known_offset():
    r = _robot("showcase6")
    rng = np.random.default_rng(11)
    delta = [0.03, -0.04, 0.02, 0.01, 0.05, -0.03]
    obs = []
    for _ in range(16):
        q = rng.uniform(-0.8, 0.8, r.ndof).tolist()
        Tk = np.array(r.fk([q[i] + delta[i] for i in range(r.ndof)]))  # row-major
        obs.append((q, Tk.T.tolist()))  # col-major
    res = caliper.calibrate_joint_offsets(r, obs)
    assert res["converged"]
    err = max(abs(res["offsets"][i] - delta[i]) for i in range(r.ndof))
    assert err < 1e-6, f"offset recovery error {err:.2e}"
    assert res["rms_residual"] < 1e-8
