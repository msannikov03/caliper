"""Phase-5 oracle: exercise the control loop, safety monitor, teleop, and
collision through the Python face. These validate the bindings + behavior; the
LeRobotDataset schema cross-check lives in test_lerobot_dataset.py."""

import math
import pathlib

import pytest

import caliper

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"


def urdf(n):
    return str(ROBOTS / f"{n}.urdf")


_REQUIRED = ("ControlLoop", "SafetyMonitor", "CollisionModel", "LeaderFollower")
pytestmark = pytest.mark.skipif(
    not all(hasattr(caliper, c) for c in _REQUIRED),
    reason="caliper lacks Phase-5 bindings — rebuild (maturin develop)",
)


# ---------- control ----------

def test_control_converges_under_gravity():
    r = caliper.Robot.from_urdf(urdf("showcase6"))
    cl = caliper.ControlLoop(r, dt=1e-3, kp=100.0, kd=20.0)
    goal = [0.2, -0.1, 0.3, 0.0, 0.1, 0.0]
    cl.run_to(goal, 8000)
    qe = math.sqrt(sum((a - b) ** 2 for a, b in zip(cl.q, goal)))
    ve = math.sqrt(sum(v * v for v in cl.qd))
    assert qe < 1e-2, f"position error {qe:.3e}"
    assert ve < 5e-2, f"velocity error {ve:.3e}"


def test_control_deterministic():
    # two identical rollouts → identical recorded frames (no wall-clock)
    def roll():
        r = caliper.Robot.from_urdf(urdf("dyn_pendulum2"))
        cl = caliper.ControlLoop(r, dt=1e-3, start=[0.5, -0.2])
        return cl.rollout_to([0.1, 0.0], 500)

    t1, s1, a1 = roll()
    t2, s2, a2 = roll()
    assert t1 == t2 and s1 == s2 and a1 == a2


def test_control_rejects_no_inertia():
    r = caliper.Robot.from_urdf(urdf("toy"))  # no <inertial>
    with pytest.raises(ValueError):
        caliper.ControlLoop(r)


# ---------- safety ----------

def test_safety_clamps_position():
    r = caliper.Robot.from_urdf(urdf("showcase6"))
    sm = caliper.SafetyMonitor(r, [0.0] * 6, dt=1.0)
    safe, v = sm.gate([100.0] * 6, )
    assert v["clamped_position"]
    assert all(abs(x) < 100.0 for x in safe)  # clamped to joint limits


def test_safety_estop_latches():
    r = caliper.Robot.from_urdf(urdf("dyn_pendulum2"))
    sm = caliper.SafetyMonitor(r, [0.3, -0.3], dt=1e-3)
    sm.estop()
    assert sm.is_estopped
    safe, v = sm.gate([0.5, 0.5], )
    assert v["estopped"]
    assert safe == [0.3, -0.3]  # held, no motion
    sm.clear_estop()
    assert not sm.is_estopped


# ---------- teleop ----------

def test_teleop_leader_follower_tracks():
    r = caliper.Robot.from_urdf(urdf("showcase6"))
    lf = caliper.LeaderFollower(r, dt=1e-3)
    # gentle leader sweep from zero → follower (rate-limited) catches up & tracks
    worst_tail = 0.0
    fq = [0.0] * 6
    for k in range(3000):
        t = k * 1e-3
        lead = [0.3 * math.sin(0.5 * t * (1 + 0.2 * i)) for i in range(6)]
        fq = lf.step(lead)
        if k > 1500:  # steady state
            worst_tail = max(worst_tail, max(abs(a - b) for a, b in zip(fq, lead)))
    assert worst_tail < 5e-3, f"steady tracking error {worst_tail:.3e}"


# ---------- collision ----------

def test_collision_self_folded_vs_extended():
    arm = caliper.Robot.from_urdf(urdf("collide_arm"))
    cm = caliper.CollisionModel(arm)
    folded = cm.query([0.0, math.pi, math.pi])
    extended = cm.query([0.0, 0.0, 0.0])
    assert folded["collision"] and folded["self_pairs"]
    assert not extended["collision"]


def test_collision_ground_and_box():
    arm = caliper.Robot.from_urdf(urdf("collide_arm"))
    cm_g = caliper.CollisionModel(arm, ground=-0.05)
    assert not cm_g.query([0.0, 0.0, 0.0])["collision"]  # upright clear
    assert cm_g.query([math.pi, 0.0, 0.0])["collision"]  # pointing down hits

    cm_b = caliper.CollisionModel(arm, boxes=[((0.0, 0.0, 0.15), (0.2, 0.2, 0.2))])
    assert cm_b.query([0.0, 0.0, 0.0])["collision"]  # box over l1


def test_collision_finite_guard():
    arm = caliper.Robot.from_urdf(urdf("collide_arm"))
    cm = caliper.CollisionModel(arm)
    with pytest.raises(ValueError):
        cm.query([0.0, float("nan"), 0.0])
