"""Phase-4 oracle (centroidal): cross-validate Caliper's mass-weighted center of
mass and total mass against Pinocchio. Both sides sum only the movable joint
subtrees — Pinocchio's `centerOfMass` excludes the universe (fixed-base) body,
and Caliper drops the fixed base's inertia the same way — so the world COM and
the mechanism mass match elementwise. This genuinely EXTENDS the externally
validated set (FK / Jacobian / RNEA / CRBA / forward-dynamics) with centroidal
quantities."""

import hashlib
import pathlib

import numpy as np
import pytest

import caliper

pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"
FIXTURES = ["dyn_pendulum2", "showcase6", "dyn_welded"]
ATOL = 1e-9
N_SAMPLES = 50
SEED = 0xCA11

_REQUIRED = ("center_of_mass", "total_mass", "has_inertia")


def _have():
    return all(hasattr(caliper.Robot, a) for a in _REQUIRED)


pytestmark = pytest.mark.skipif(
    not _have(),
    reason="caliper.Robot lacks centroidal bindings — rebuild (maturin develop)",
)


def _urdf(n):
    return str(ROBOTS / f"{n}.urdf")


def _rng(n):
    tag = int.from_bytes(hashlib.sha256(n.encode()).digest()[:4], "little")
    return np.random.default_rng([SEED, tag])


def _load(name):
    model = pin.buildModelFromUrdf(_urdf(name))
    data = model.createData()
    robot = caliper.Robot.from_urdf(_urdf(name))
    assert model.nq == robot.ndof and model.nv == robot.ndof, f"[{name}] multi-DoF joint"
    assert robot.has_inertia, f"[{name}] caliper reports no inertia (fixture missing <inertial>?)"
    return model, data, robot


def _scatter_q(model, robot, q):
    out = np.zeros(model.nq)
    for i, name in enumerate(robot.joint_names):
        out[model.joints[model.getJointId(name)].idx_q] = q[i]
    return out


def _sample_q(rng, model, robot):
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
    return q


@pytest.mark.parametrize("name", FIXTURES)
def test_total_mass_matches_pinocchio(name):
    model, _, robot = _load(name)
    # Pinocchio's movable-body masses (idx 0 is the universe / fixed base — excluded).
    mass_pin = sum(model.inertias[j].mass for j in range(1, model.njoints))
    assert abs(robot.total_mass - mass_pin) <= ATOL, (
        f"[{name}] total_mass {robot.total_mass} != pin movable mass {mass_pin}"
    )


@pytest.mark.parametrize("name", FIXTURES)
def test_center_of_mass_matches_pinocchio(name):
    model, data, robot = _load(name)
    rng = _rng(name)
    ran = 0
    for _ in range(N_SAMPLES):
        q = _sample_q(rng, model, robot)
        com_pin = np.asarray(pin.centerOfMass(model, data, _scatter_q(model, robot, q)))
        com_cal = np.asarray(robot.center_of_mass(list(q)))
        w = float(np.abs(com_cal - com_pin).max())
        assert w <= ATOL + 1e-9 * max(1.0, np.abs(com_pin).max()), (
            f"[{name}] COM worst |d|={w:.3e}\nq={q.tolist()}\n"
            f"caliper={com_cal.tolist()} pin={com_pin.tolist()}"
        )
        ran += 1
    assert ran == N_SAMPLES


@pytest.mark.parametrize("name", FIXTURES)
def test_total_mass_is_config_invariant(name):
    # total_mass is a static mechanism property; querying COM at varied q must
    # not perturb it (guards against an accidental q-dependence bug).
    model, _, robot = _load(name)
    rng = _rng(name)
    m0 = robot.total_mass
    for _ in range(5):
        robot.center_of_mass(list(_sample_q(rng, model, robot)))
        assert robot.total_mass == m0
