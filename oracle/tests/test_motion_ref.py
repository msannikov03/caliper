"""Cross-validate caliper's jerk-limited move_j against RUCKIG, the
gold-standard independent online-trajectory-generation library.

The Rust suite in caliper-motion and oracle/test_motion.py are self-consistent
(they re-diff caliper's own output). This module closes that gap: it builds a
RUCKIG trajectory with the *same* vmax/amax/jmax and checks that

  1. caliper's time-synchronized rest-to-rest duration is >= RUCKIG's per-dof
     time-optimal duration (you cannot beat the slowest dof's optimum) and
     matches RUCKIG's synchronized minimum time within a sensible tolerance;
  2. caliper's sampled trajectory is feasible — |qd|<=vmax, |qdd|<=amax, and a
     finite-difference |jerk|<=jmax — re-checked independently in numpy;
  3. endpoints are exact and the move is rest-to-rest (qd=qdd=0 at both ends).

RUCKIG models the rest-to-rest jerk-limited case exactly, so it is the primary
reference; no numpy ODE fallback is needed here.
"""

import pathlib

import pytest

caliper = pytest.importorskip("caliper")
np = pytest.importorskip("numpy")
ruckig = pytest.importorskip("ruckig")

from ruckig import InputParameter, Result, Ruckig, Trajectory  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"

# Skip cleanly until the bindings ship move_j / Trajectory (mirrors test_motion).
_probe = caliper.Robot.from_urdf(str(ROBOTS / "showcase6.urdf"))
if not hasattr(_probe, "move_j"):
    pytest.skip("caliper.Robot.move_j not built yet", allow_module_level=True)

# Tolerances. caliper's S-curve planner and RUCKIG are both analytic
# time-optimal jerk-limited generators, so the synchronized durations should
# agree tightly; these bounds still catch gross (e.g. 2x) regressions.
_DUR_RTOL = 0.05
_DUR_ATOL = 1e-2
_LIM_RTOL = 1e-3  # caliper samples are analytic -> near-exact at the bound
_JERK_RTOL = 5e-2  # finite-difference of a piecewise-linear qdd


def _robot():
    return caliper.Robot.from_urdf(str(ROBOTS / "showcase6.urdf"))


def _ruckig_duration(q0, q1, vmax, amax, jmax):
    """RUCKIG's time-synchronized minimum duration + per-dof optimal times."""
    n = len(q0)
    otg = Ruckig(n)
    inp = InputParameter(n)
    inp.current_position = list(q0)
    inp.current_velocity = [0.0] * n
    inp.current_acceleration = [0.0] * n
    inp.target_position = list(q1)
    inp.target_velocity = [0.0] * n
    inp.target_acceleration = [0.0] * n
    inp.max_velocity = list(vmax)
    inp.max_acceleration = list(amax)
    inp.max_jerk = list(jmax)
    traj = Trajectory(n)
    res = otg.calculate(inp, traj)
    # Offline calculate() returns Working on success; Error.* values are < 0.
    assert res in (Result.Working, Result.Finished), f"ruckig failed: {res}"
    return traj.duration, list(traj.independent_min_durations), traj


def _check_against_ruckig(q0, q1, label):
    r = _robot()
    traj = r.move_j(list(q0), list(q1))
    vmax = np.array(traj.vel_limit)
    amax = np.array(traj.accel_limit)
    jmax = np.array(traj.jerk_limit)

    # (1) duration vs RUCKIG -------------------------------------------------
    rk_dur, rk_indep, _ = _ruckig_duration(q0, q1, vmax, amax, jmax)
    slowest = max(rk_indep)
    cal_dur = traj.duration
    # cannot be faster than the slowest dof's time-optimal solo move
    assert cal_dur >= slowest - _DUR_ATOL, (
        f"[{label}] caliper {cal_dur:.6f}s faster than slowest-dof optimum "
        f"{slowest:.6f}s"
    )
    # must match RUCKIG's synchronized minimum time
    assert cal_dur == pytest.approx(rk_dur, rel=_DUR_RTOL, abs=_DUR_ATOL), (
        f"[{label}] caliper {cal_dur:.6f}s vs ruckig {rk_dur:.6f}s"
    )

    # (2) feasibility, independently re-checked ------------------------------
    t, q, qd, qdd = traj.sample_uniform(0.002)
    t = np.asarray(t)
    q = np.asarray(q)
    qd = np.asarray(qd)
    qdd = np.asarray(qdd)
    assert np.all(np.abs(qd) <= vmax * (1.0 + _LIM_RTOL) + 1e-9), f"[{label}] vmax violated"
    assert np.all(np.abs(qdd) <= amax * (1.0 + _LIM_RTOL) + 1e-9), f"[{label}] amax violated"
    # finite-difference jerk = d(qdd)/dt. qdd is continuous piecewise-linear,
    # so its numerical gradient stays within the analytic jerk band.
    if len(t) >= 3:
        dt = t[1] - t[0]
        jerk = np.gradient(qdd, dt, axis=0)
        assert np.all(np.abs(jerk) <= jmax * (1.0 + _JERK_RTOL) + 1e-6), (
            f"[{label}] jmax violated (fd jerk max "
            f"{np.max(np.abs(jerk) / jmax):.3f}x)"
        )

    # (3) endpoints exact + rest --------------------------------------------
    assert np.allclose(q[0], q0, atol=1e-9), f"[{label}] start position drift"
    assert np.allclose(q[-1], q1, atol=1e-6), f"[{label}] end position drift"
    assert np.allclose(qd[0], 0.0, atol=1e-9), f"[{label}] nonzero start velocity"
    assert np.allclose(qd[-1], 0.0, atol=1e-6), f"[{label}] nonzero end velocity"
    assert np.allclose(qdd[0], 0.0, atol=1e-9), f"[{label}] nonzero start accel"
    assert np.allclose(qdd[-1], 0.0, atol=1e-6), f"[{label}] nonzero end accel"
    assert traj.completed


def test_move_j_single_dof_vs_ruckig():
    """Move only joint 0; duration is set by that one dof."""
    n = _robot().ndof
    q0 = [0.0] * n
    q1 = [0.5] + [0.0] * (n - 1)
    _check_against_ruckig(q0, q1, "single-dof")


def test_move_j_multi_dof_vs_ruckig():
    """All six joints move different distances -> time synchronization."""
    n = _robot().ndof
    q0 = [0.0] * n
    q1 = [0.5, -0.3, 0.4, 0.2, -0.6, 0.1][:n]
    _check_against_ruckig(q0, q1, "multi-dof")


def test_ruckig_slowest_dof_drives_sync_duration():
    """The synchronized duration equals the slowest dof's independent optimum
    (sanity check on the reference itself, then tied back to caliper)."""
    n = _robot().ndof
    q0 = [0.0] * n
    q1 = [0.5, -0.3, 0.4, 0.2, -0.6, 0.1][:n]
    r = _robot()
    traj = r.move_j(q0, q1)
    rk_dur, rk_indep, _ = _ruckig_duration(
        q0, q1, traj.vel_limit, traj.accel_limit, traj.jerk_limit
    )
    assert rk_dur == pytest.approx(max(rk_indep), rel=1e-6, abs=1e-9)
    assert traj.duration == pytest.approx(rk_dur, rel=_DUR_RTOL, abs=_DUR_ATOL)
