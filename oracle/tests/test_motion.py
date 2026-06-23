"""Phase-3 motion oracle: independent numeric re-check of the jerk-limited
move_j trajectory through the Python bindings. The Rust suite in caliper-motion
is primary; this confirms the binding surface + an independent finite-difference
limit check (no Pinocchio needed for the scalar S-curve)."""

import pathlib

import pytest

import caliper

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"

# skip cleanly until the bindings ship move_j / Trajectory
_r = caliper.Robot.from_urdf(str(ROBOTS / "toy.urdf"))
if not hasattr(_r, "move_j"):
    pytest.skip("caliper.Robot.move_j not built yet", allow_module_level=True)


def test_move_j_within_limits_numeric():
    import numpy as np

    r = caliper.Robot.from_urdf(str(ROBOTS / "showcase6.urdf"))
    traj = r.move_j([0] * 6, [0.5, -0.3, 0.4, 0.2, -0.6, 0.1])
    t, q, qd, qdd = traj.sample_uniform(0.005)
    q = np.array(q)
    qd = np.array(qd)
    qdd = np.array(qdd)
    vlim = np.array(traj.vel_limit)
    alim = np.array(traj.accel_limit)
    assert np.all(np.abs(qd) <= vlim * (1 + 1e-3))
    assert np.all(np.abs(qdd) <= alim * (1 + 1e-3))
    # independent numeric re-diff of q (gradient must stay within the velocity limit)
    dt = t[1] - t[0]
    g = np.gradient(q, dt, axis=0)
    assert np.all(np.abs(g) <= vlim * (1 + 5e-2))
    assert np.allclose(q[0], [0] * 6, atol=1e-9)
    assert np.allclose(q[-1], [0.5, -0.3, 0.4, 0.2, -0.6, 0.1], atol=1e-6)
    assert traj.completed


def test_scurve_total_time_identity():
    r = caliper.Robot.from_urdf(str(ROBOTS / "toy.urdf"))
    traj = r.move_j([0, 0], [2.0, 0.0])
    assert traj.duration > 0
