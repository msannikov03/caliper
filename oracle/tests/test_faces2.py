"""Cross-validate the faces-completeness bindings (TOPP retiming, PRM,
penetration contacts, redundancy resolution, streaming control) end-to-end
through the Python face."""

import math

import numpy as np
import pytest

caliper = pytest.importorskip("caliper")

FIX = "oracle/fixtures/robots"


def _robot(name):
    return caliper.Robot.from_urdf(f"{FIX}/{name}.urdf")


# ---------- TOPP retime_time_optimal ----------
def test_retime_time_optimal_respects_limits_and_hits_waypoints():
    r = _robot("showcase6")
    wps = [
        [0.0] * 6,
        [0.4, -0.3, 0.2, 0.1, -0.2, 0.3],
        [-0.2, 0.1, -0.1, 0.3, 0.2, -0.1],
    ]
    vmax, amax = [1.0] * 6, [2.0] * 6
    traj = r.retime_time_optimal(wps, vmax, amax)
    assert traj.duration > 0.0
    assert all(math.isinf(j) for j in traj.jerk_limit)  # bang-bang: jerk unbounded
    _, q, qd, qdd = traj.sample_uniform(1e-3)
    q, qd, qdd = np.array(q), np.array(qd), np.array(qdd)
    tol = 1.0 + 1e-6
    assert (np.abs(qd) <= np.array(vmax) * tol).all(), "vmax violated"
    assert (np.abs(qdd) <= np.array(amax) * tol).all(), "amax violated"
    # endpoints exact, interior waypoint hit at a corner stop
    assert np.allclose(traj.q_at(0.0), wps[0], atol=1e-9)
    assert np.allclose(traj.q_at(traj.duration), wps[-1], atol=1e-9)
    for wp in wps:
        d = np.abs(q - np.array(wp)).max(axis=1).min()
        assert d < 1e-4, f"waypoint {wp} missed by {d:.2e}"


def test_retime_time_optimal_default_limits_and_arg_guard():
    r = _robot("showcase6")
    traj = r.retime_time_optimal([[0.0] * 6, [0.3] * 6])  # model-default limits
    assert traj.duration > 0.0
    with pytest.raises(ValueError):
        r.retime_time_optimal([[0.0] * 6, [0.3] * 6], vmax=[1.0] * 6)  # amax missing
    with pytest.raises(ValueError):
        r.retime_time_optimal([[0.0] * 5, [0.3] * 5], [1.0] * 6, [2.0] * 6)


# ---------- PRM plan_prm ----------
def test_plan_prm_deterministic_collision_free():
    r = _robot("collide_arm")
    ground, boxes = -0.1, [((0.6, 0.0, 0.3), (0.15, 0.15, 0.15))]
    cm = caliper.CollisionModel(r, ground=ground, boxes=boxes)
    rng = np.random.default_rng(5)

    def sample_free():
        for _ in range(500):
            q = [float(x) for x in rng.uniform(-1.0, 1.0, r.ndof)]
            if not cm.query(q)["collision"]:
                return q
        pytest.skip("could not sample a free config")

    start, goal = sample_free(), sample_free()
    p1 = caliper.Planner(r, ground=ground, boxes=boxes, seed=7).plan_prm(start, goal, 600, 10)
    p2 = caliper.Planner(r, ground=ground, boxes=boxes, seed=7).plan_prm(start, goal, 600, 10)
    assert p1 == p2, "same seed/samples/k must give the same path"
    assert np.allclose(p1[0], start) and np.allclose(p1[-1], goal)
    P = caliper.Planner(r, ground=ground, boxes=boxes, seed=7)
    assert P.verify(p1), "PRM path failed independent collision re-verification"


def test_plan_prm_rejects_zero_budget():
    r = _robot("collide_arm")
    P = caliper.Planner(r)
    z = [0.0] * r.ndof
    with pytest.raises(ValueError):
        P.plan_prm(z, z, 0, 8)
    with pytest.raises(ValueError):
        P.plan_prm(z, z, 100, 0)


# ---------- penetration contacts ----------
def test_contacts_on_folded_arm_sane_depth_and_normal():
    arm = _robot("collide_arm")
    cm = caliper.CollisionModel(arm)
    folded = cm.contacts([0.0, math.pi, math.pi])  # known self-collision
    assert folded, "folded arm must report at least one contact"
    pairs = set(map(tuple, cm.query([0.0, math.pi, math.pi])["self_pairs"]))
    for a, b, c in folded:
        assert a < b and (a, b) in pairs
        assert c["depth"] > 0.0
        assert abs(np.linalg.norm(c["normal"]) - 1.0) < 1e-9
        assert len(c["witness"]) == 3 and np.isfinite(c["witness"]).all()
    assert cm.contacts([0.0, 0.0, 0.0]) == []  # extended arm is contact-free


# ---------- redundancy resolution ----------
def test_nullspace_step_moves_joints_but_not_tip():
    r = _robot("redundant7")
    q = [0.3, 0.2, -0.4, 0.5, 0.3, -0.2, 0.4]
    z = [0.2, -0.1, 0.15, 0.1, -0.2, 0.05, 0.1]
    qd = r.nullspace_step(q, [0.0] * 6, z)
    assert np.linalg.norm(qd) > 1e-8, "no null-space motion on a redundant arm"
    eps = 1e-6
    T0 = np.array(r.fk(q))
    T1 = np.array(r.fk((np.array(q) + eps * np.array(qd)).tolist()))
    assert np.abs(T1 - T0).max() < 1e-8, "null-space step moved the tip"


def test_resolved_rate_tracks_desired_tip_velocity():
    r = _robot("redundant7")
    q = [0.3, 0.2, -0.4, 0.5, 0.3, -0.2, 0.4]
    v = [0.01, -0.02, 0.015, 0.0, 0.005, -0.01]
    qd = r.resolved_rate(q, v)
    J = np.array(r.jacobian(q))
    assert np.allclose(J @ np.array(qd), v, atol=1e-6)
    with pytest.raises(ValueError):
        r.resolved_rate(q, [0.0] * 5)  # v must be length 6


# ---------- streaming control ----------
def test_run_stream_matches_rollout_bitwise():
    def loop():
        r = _robot("dyn_pendulum2")
        return caliper.ControlLoop(r, dt=1e-3, start=[0.5, -0.2])

    goal, ticks = [0.1, 0.0], 400
    times, states, actions = loop().rollout_to(goal, ticks)
    streamed = []
    cl = loop()
    executed = cl.run_stream(goal, ticks, lambda f: streamed.append((f["t"], f["measured"], f["command"])))
    assert executed == ticks
    assert [s[0] for s in streamed] == times
    assert [s[1] for s in streamed] == states  # bitwise: engine guarantees identity
    assert [s[2] for s in streamed] == actions


def test_run_stream_decimation_and_cancel():
    def loop():
        r = _robot("dyn_pendulum2")
        return caliper.ControlLoop(r, dt=1e-3, start=[0.5, -0.2])

    goal, ticks = [0.1, 0.0], 100
    full, dec = [], []
    loop().run_stream(goal, ticks, lambda f: full.append(f["tick"]))
    loop().run_stream(goal, ticks, lambda f: dec.append(f["tick"]), emit_every=7)
    assert dec == full[::7], "decimation must be a strict subsample"
    # returning False cancels cooperatively at that tick
    seen = []

    def cancel_after_three(f):
        seen.append(f["tick"])
        return len(seen) < 3

    executed = loop().run_stream(goal, ticks, cancel_after_three)
    assert executed == 3 and len(seen) == 3
