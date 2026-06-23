"""Phase-4 oracle: cross-validate Caliper dynamics (RNEA / CRBA / forward dynamics)
against Pinocchio. Gravity is pinned IDENTICAL on both sides ([0,0,-9.81]) plus a
zero-g control case. The dyn_welded fixture exercises the fixed-link inertia FOLD
(Pinocchio folds fixed joints the same way), the one path showcase6 cannot test."""

import hashlib
import pathlib

import numpy as np
import pytest

import caliper

pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"
FIXTURES_DYN = ["dyn_pendulum2", "showcase6", "dyn_welded"]
ATOL = 1e-9
N_SAMPLES = 50
SEED = 0xCA11
GRAVITY = np.array([0.0, 0.0, -9.81])

_REQUIRED = ("rnea", "crba", "forward_dynamics", "gravity_torque", "has_inertia")


def _have():
    return all(hasattr(caliper.Robot, a) for a in _REQUIRED)


pytestmark = pytest.mark.skipif(
    not _have(),
    reason="caliper.Robot lacks dynamics bindings — rebuild (maturin develop) after Phase 4",
)


def _urdf(n):
    return str(ROBOTS / f"{n}.urdf")


def _rng(n):
    tag = int.from_bytes(hashlib.sha256(n.encode()).digest()[:4], "little")
    return np.random.default_rng([SEED, tag])


def _load(name):
    model = pin.buildModelFromUrdf(_urdf(name))
    model.gravity.linear = GRAVITY.copy()  # pin gravity EXPLICIT (not the 9.80665 default)
    model.gravity.angular = np.zeros(3)
    data = model.createData()
    robot = caliper.Robot.from_urdf(_urdf(name))
    assert model.nq == robot.ndof and model.nv == robot.ndof, f"[{name}] multi-DoF joint"
    assert robot.has_inertia, f"[{name}] caliper reports no inertia (fixture missing <inertial>?)"
    return model, data, robot


def _scatter(model, robot, x, which):  # which='q'|'v'
    out = np.zeros(model.nq if which == "q" else model.nv)
    for i, name in enumerate(robot.joint_names):
        j = model.joints[model.getJointId(name)]
        out[(j.idx_q if which == "q" else j.idx_v)] = x[i]
    return out


def _perm(model, robot):  # pin idx_v column → caliper joint order
    return [model.joints[model.getJointId(n)].idx_v for n in robot.joint_names]


def _sample(rng, model, robot):
    n = robot.ndof
    q = np.empty(n)
    for i, name in enumerate(robot.joint_names):
        iq = model.joints[model.getJointId(name)].idx_q
        lo, hi = model.lowerPositionLimit[iq], model.upperPositionLimit[iq]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = -2.0, 2.0
        else:
            lo, hi = max(lo, -2.0), min(hi, 2.0)
        q[i] = rng.uniform(lo, hi)
    qd = rng.uniform(-1.5, 1.5, n)
    qdd = rng.uniform(-1.5, 1.5, n)
    return q, qd, qdd


def _worst(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b))
    k = int(np.argmax(d))
    return float(d.flat[k]), np.unravel_index(k, d.shape)


@pytest.mark.parametrize("name", FIXTURES_DYN)
@pytest.mark.parametrize("grav", [GRAVITY, np.zeros(3)])  # g and a zero-g control
def test_rnea_matches_pinocchio(name, grav):
    model, data, robot = _load(name)
    model.gravity.linear = grav.copy()
    perm = _perm(model, robot)
    rng = _rng(name)
    ran = 0
    loaded = False
    for _ in range(N_SAMPLES):
        q, qd, qdd = _sample(rng, model, robot)
        tau_pin = np.asarray(
            pin.rnea(
                model,
                data,
                _scatter(model, robot, q, "q"),
                _scatter(model, robot, qd, "v"),
                _scatter(model, robot, qdd, "v"),
            )
        )[perm]
        tau_cal = np.asarray(robot.rnea(list(q), list(qd), list(qdd), gravity=list(grav)))
        if np.abs(tau_cal).max() > 1e-3:
            loaded = True
        w, where = _worst(tau_cal, tau_pin)
        assert w <= ATOL + 1e-9 * max(1.0, np.abs(tau_pin).max()), (
            f"[{name}] RNEA worst |d|={w:.3e} at {where}\nq={q.tolist()} g={grav.tolist()}"
        )
        ran += 1
    assert ran == N_SAMPLES
    if np.any(grav):
        assert loaded, f"[{name}] RNEA sweep never loaded the arm"


@pytest.mark.parametrize("name", FIXTURES_DYN)
def test_crba_matches_pinocchio(name):
    model, data, robot = _load(name)
    perm = _perm(model, robot)
    rng = _rng(name)
    ran = 0
    for _ in range(N_SAMPLES):
        q, _, _ = _sample(rng, model, robot)
        M_pin = np.asarray(pin.crba(model, data, _scatter(model, robot, q, "q")))
        M_pin = np.triu(M_pin)
        M_pin = M_pin + M_pin.T - np.diag(np.diag(M_pin))  # pin fills upper only
        M_pin = M_pin[np.ix_(perm, perm)]
        M_cal = np.asarray(robot.crba(list(q)))
        assert np.allclose(M_cal, M_cal.T, atol=1e-12), f"[{name}] M not symmetric"
        assert np.linalg.eigvalsh(M_cal).min() > 0, f"[{name}] M not SPD"
        w, where = _worst(M_cal, M_pin)
        assert w <= ATOL + 1e-9 * max(1.0, np.abs(M_pin).max()), (
            f"[{name}] CRBA worst |d|={w:.3e} at {where}\nq={q.tolist()}"
        )
        ran += 1
    assert ran == N_SAMPLES


@pytest.mark.parametrize("name", FIXTURES_DYN)
def test_forward_dynamics_vs_aba_and_roundtrip(name):
    model, data, robot = _load(name)
    perm = _perm(model, robot)
    rng = _rng(name)
    ran = 0
    for _ in range(N_SAMPLES):
        q, qd, _ = _sample(rng, model, robot)
        tau = rng.uniform(-2.0, 2.0, robot.ndof)
        # independent FD path: Pinocchio ABA
        a_pin = np.asarray(
            pin.aba(
                model,
                data,
                _scatter(model, robot, q, "q"),
                _scatter(model, robot, qd, "v"),
                _scatter(model, robot, tau, "v"),
            )
        )[perm]
        a_cal = np.asarray(
            robot.forward_dynamics(list(q), list(qd), list(tau), gravity=list(GRAVITY))
        )
        w, _ = _worst(a_cal, a_pin)
        assert w <= ATOL + 1e-9 * max(1.0, np.abs(a_pin).max()), f"[{name}] FD vs ABA {w:.3e}"
        # self-consistency round-trip: tau=ID(qdd) -> FD recovers qdd
        qdd = rng.uniform(-1.5, 1.5, robot.ndof)
        t2 = robot.rnea(list(q), list(qd), list(qdd), gravity=list(GRAVITY))
        qdd2 = robot.forward_dynamics(list(q), list(qd), list(t2), gravity=list(GRAVITY))
        assert np.allclose(qdd, qdd2, atol=1e-9), f"[{name}] FD round-trip"
        ran += 1
    assert ran == N_SAMPLES


def test_simulator_energy_bounded():
    if not hasattr(caliper, "Simulator"):
        pytest.skip("Simulator binding")
    r = caliper.Robot.from_urdf(_urdf("dyn_pendulum2"))
    # substeps=16 → integrator h≈6e-5. Semi-implicit Euler conserves energy only to
    # a BOUNDED O(h) oscillation (the symplectic property; a leaky integrator would
    # drift secularly) — ~0.6% here for this vigorous passive swing.
    sim = caliper.Simulator(r, dt=1e-3, gravity=[0, 0, -9.81], damping=0.0, substeps=16)
    sim.reset(q0=[1.0, 0.3])
    e0 = sim.energy
    worst = 0.0
    for _ in range(2000):
        sim.step()
        worst = max(worst, abs(sim.energy - e0) / abs(e0))
    assert max(abs(sim.q[0] - 1.0), abs(sim.q[1] - 0.3)) > 0.1
    assert worst < 1e-2
