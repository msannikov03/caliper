"""Phase-1 external oracle: cross-validate Caliper's FK and geometric Jacobian
against Pinocchio (the reference rigid-body-dynamics library).

This is the *independent* check on the engine's kinematics: Pinocchio parses the
SAME URDF and computes FK / Jacobians with a completely different code path, so
element-wise agreement is strong evidence the math is right (not just internally
self-consistent, which the Rust finite-difference tests already cover).

Conventions reconciled here (verified against the source, not assumed):
  * Spatial-velocity ordering is ``[v; w]`` (linear first) on BOTH sides -- Caliper
    stores twists ``[v; w]`` to match Pinocchio's ``Motion`` -- so NO row swap.
  * Caliper ``jacobian(..., "world")`` == Pinocchio ``LOCAL_WORLD_ALIGNED``
    (axes world-aligned, reference point at the frame origin).
    Caliper ``jacobian(..., "body")``  == Pinocchio ``LOCAL``.
    Pinocchio ``WORLD`` is the spatial velocity referred to the world ORIGIN -- NOT
    what we want -- and is deliberately not used.
  * Caliper ``fk(q, name)`` (world pose of the named link frame) ==
    ``data.oMf[getFrameId(name)]``. A URDF link frame is its parent joint's child
    frame, so a Pinocchio BODY frame named after the link coincides with our frame
    of the same name -- including links reached through fixed joints we fold away.

Joint-order handling (the subtle part):
  * Caliper's ``q`` is ordered by ``robot.joint_names`` (topo-sorted movable joints).
    Pinocchio has its OWN joint order. We therefore map q BY JOINT NAME into
    Pinocchio's configuration vector using each joint's ``idx_q``.
  * Pinocchio's Jacobian COLUMNS are in Pinocchio joint order (idx_v); we reorder
    them to Caliper joint order by name using each joint's ``idx_v``. (idx_q indexes
    the configuration q; idx_v indexes the tangent/velocity -- kept distinct.)
  * Fixtures are revolute/prismatic only (1 DoF, nq == nv == ndof). We assert
    ``model.nq == ndof`` and ``model.nv == ndof`` up front so any ``continuous``
    joint (a 2-D cos/sin pair in Pinocchio) trips the test loudly instead of
    silently corrupting the by-name scalar mapping.

A dedicated log6 round-trip (``pin.log6(M)`` vs Caliper's twist) is intentionally
NOT compared here: the binding does not expose log6, and FK/Jacobian element-wise
agreement is the load-bearing signal. The Rust suite covers log6 self-consistency
separately, so this oracle stays scoped to (R, t) poses and the geometric Jacobian.

This module is skipped wholesale when Pinocchio is not importable (it is an
optional dev dependency), and skipped per-module when the Caliper Python bindings
have not yet been extended with ``fk`` / ``jacobian`` / ``tip_frame`` /
``frame_names``.
"""
import hashlib
import math
import pathlib

import numpy as np
import pytest

import caliper

# Optional dependency: skip the whole module cleanly if Pinocchio is absent.
pin = pytest.importorskip("pinocchio", reason="pinocchio (pin) not installed")

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"

# Fixtures with at least one movable joint. (fixed_only / disconnected excluded:
# 0-DOF has no q to sweep and nothing to compare a Jacobian against.)
FIXTURES = ["toy", "prismatic", "branched", "redundant7"]

ATOL = 1e-9
N_SAMPLES = 50
SEED = 0xCA11  # deterministic, fixture-independent

# The Caliper Robot bindings must expose these for the cross-check to run. Until
# the PyO3 surface is extended (Phase-1 Step 7) we skip rather than error, so the
# suite stays green on a not-yet-rebuilt extension.
_REQUIRED = ("fk", "jacobian", "tip_frame", "frame_names")


def _have_bindings():
    return all(hasattr(caliper.Robot, attr) for attr in _REQUIRED)


pytestmark = pytest.mark.skipif(
    not _have_bindings(),
    reason="caliper.Robot lacks fk/jacobian/tip_frame/frame_names -- rebuild "
    "bindings (maturin develop) after Phase-1 Step 7",
)


def _urdf(name):
    return str(ROBOTS / f"{name}.urdf")


def _load(name):
    """Build the Pinocchio (fixed-base) model+data and the Caliper robot."""
    path = _urdf(name)
    model = pin.buildModelFromUrdf(path)
    data = model.createData()
    robot = caliper.Robot.from_urdf(path)
    return model, data, robot


def _q_to_pinocchio(model, robot, q):
    """Scatter Caliper q (ordered by robot.joint_names) into a Pinocchio config
    vector by JOINT NAME via each joint's idx_q. Requires nq == ndof (1-DoF
    joints only) -- asserted by the caller."""
    q_pin = np.zeros(model.nq)
    for i, name in enumerate(robot.joint_names):
        jid = model.getJointId(name)  # 1-based; 0 == "universe"
        q_pin[model.joints[jid].idx_q] = q[i]
    return q_pin


def _column_perm_pin_to_caliper(model, robot):
    """For each Caliper joint i (in robot.joint_names order) the Pinocchio Jacobian
    column index (idx_v) holding that joint's tangent. Lets us reorder Pinocchio's
    columns into Caliper's column order before an element-wise compare."""
    return [model.joints[model.getJointId(name)].idx_v for name in robot.joint_names]


def _rng(name):
    # Per-fixture stream so a failure in one fixture is reproducible in isolation.
    # Use a stable digest (not the salt-randomized builtin hash) so the sweep is
    # byte-for-byte reproducible across processes.
    tag = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return np.random.default_rng([SEED, tag])


def _sample_q(rng, model, robot):
    """A random config in joint limits where finite, else in [-2, 2]. Returned in
    Caliper joint order."""
    ndof = robot.ndof
    q = np.empty(ndof)
    for i, name in enumerate(robot.joint_names):
        jid = model.getJointId(name)
        iq = model.joints[jid].idx_q
        lo = model.lowerPositionLimit[iq]
        hi = model.upperPositionLimit[iq]
        # Pinocchio uses +/-inf for unbounded joints; fall back to a generic range.
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = -2.0, 2.0
        else:
            lo, hi = max(lo, -2.0), min(hi, 2.0)
            if hi <= lo:  # pathological clamp -- widen back out
                lo, hi = -2.0, 2.0
        q[i] = rng.uniform(lo, hi)
    return q


def _worst(a, b):
    """(max abs diff, flat index) between two same-shaped arrays."""
    diff = np.abs(np.asarray(a) - np.asarray(b))
    idx = int(np.argmax(diff))
    return float(diff.flat[idx]), np.unravel_index(idx, diff.shape)


@pytest.mark.parametrize("name", FIXTURES)
def test_fk_matches_pinocchio(name):
    model, data, robot = _load(name)
    # Guard: by-name scalar mapping assumes one config coord per movable joint.
    assert model.nq == robot.ndof, (
        f"[{name}] nq ({model.nq}) != ndof ({robot.ndof}) -- a multi-DoF joint "
        "(e.g. 'continuous') broke the 1:1 idx_q mapping"
    )

    rng = _rng(name)
    compared = 0  # non-vacuity guard: at least one real comparison must run
    moved = False  # at least one compared frame must be non-identity
    for s in range(N_SAMPLES):
        q = _sample_q(rng, model, robot)
        q_pin = _q_to_pinocchio(model, robot, q)
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)

        for fname in robot.frame_names():
            # Only frames Pinocchio also exposes (it does not necessarily mint a
            # frame for every URDF link the way we do; skip ones it lacks).
            if not model.existFrame(fname):
                continue
            fid = model.getFrameId(fname)
            T_pin = np.asarray(data.oMf[fid].homogeneous)
            T_cal = np.asarray(robot.fk(q, fname))
            assert T_cal.shape == (4, 4), f"[{name}] fk('{fname}') not 4x4"
            compared += 1
            if not np.allclose(T_cal, np.eye(4), atol=1e-6):
                moved = True
            wmax, where = _worst(T_cal, T_pin)
            assert wmax <= ATOL, (
                f"FK mismatch [{name}] frame='{fname}' sample={s}: "
                f"worst |delta|={wmax:.3e} at {where} (atol={ATOL})\n"
                f"q={q.tolist()}\ncaliper=\n{T_cal}\npinocchio=\n{T_pin}"
            )

    assert compared > 0, (
        f"[{name}] FK comparison was VACUOUS: zero frames matched between "
        f"caliper.frame_names()={robot.frame_names()} and the Pinocchio model"
    )
    assert moved, (
        f"[{name}] every compared FK frame was identity -- the sweep proved "
        "nothing; check frame names / q sampling"
    )


@pytest.mark.parametrize("name", FIXTURES)
def test_jacobian_matches_pinocchio(name):
    model, data, robot = _load(name)
    assert model.nq == robot.ndof, (
        f"[{name}] nq ({model.nq}) != ndof ({robot.ndof}) -- multi-DoF joint "
        "broke the 1:1 idx mapping"
    )
    assert model.nv == robot.ndof, f"[{name}] nv ({model.nv}) != ndof ({robot.ndof})"

    tip = robot.tip_frame()
    assert isinstance(tip, str), (
        f"[{name}] tip_frame() must return a frame NAME string, got {type(tip)}"
    )
    assert model.existFrame(tip), (
        f"[{name}] tip frame '{tip}' not present in Pinocchio model"
    )
    fid = model.getFrameId(tip)
    perm = _column_perm_pin_to_caliper(model, robot)

    rng = _rng(name)
    ran = 0  # non-vacuity guard: the sample loop must actually execute
    for s in range(N_SAMPLES):
        q = _sample_q(rng, model, robot)
        q_pin = _q_to_pinocchio(model, robot, q)
        pin.forwardKinematics(model, data, q_pin)
        pin.computeJointJacobians(model, data, q_pin)
        pin.updateFramePlacements(model, data)

        # ---- LOCAL_WORLD_ALIGNED  <->  caliper "world" ----------------------
        J_pin_lwa = np.asarray(
            pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        )[:, perm]
        J_cal_w = np.asarray(robot.jacobian(q, tip, "world"))
        assert J_cal_w.shape == J_pin_lwa.shape, (
            f"[{name}] world Jacobian shape {J_cal_w.shape} != {J_pin_lwa.shape}"
        )
        wmax, where = _worst(J_cal_w, J_pin_lwa)
        assert wmax <= ATOL, (
            f"Jacobian(world) mismatch [{name}] frame='{tip}' sample={s}: "
            f"worst |delta|={wmax:.3e} at row/col {where} (atol={ATOL})\n"
            f"q={q.tolist()}\ncaliper=\n{J_cal_w}\npinocchio=\n{J_pin_lwa}"
        )

        # ---- LOCAL  <->  caliper "body" ------------------------------------
        J_pin_local = np.asarray(
            pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL)
        )[:, perm]
        J_cal_b = np.asarray(robot.jacobian(q, tip, "body"))
        assert J_cal_b.shape == J_pin_local.shape, (
            f"[{name}] body Jacobian shape {J_cal_b.shape} != {J_pin_local.shape}"
        )
        wmax, where = _worst(J_cal_b, J_pin_local)
        assert wmax <= ATOL, (
            f"Jacobian(body) mismatch [{name}] frame='{tip}' sample={s}: "
            f"worst |delta|={wmax:.3e} at row/col {where} (atol={ATOL})\n"
            f"q={q.tolist()}\ncaliper=\n{J_cal_b}\npinocchio=\n{J_pin_local}"
        )
        ran += 1

    assert ran == N_SAMPLES, (
        f"[{name}] Jacobian sweep ran {ran}/{N_SAMPLES} samples -- non-vacuity "
        "guard tripped"
    )


# ===== Phase 2: singularity / manipulability cross-validation =====

_REQUIRED_ANALYZE = ("analyze", "manipulability")


def _have_analyze():
    return all(hasattr(caliper.Robot, a) for a in _REQUIRED_ANALYZE)


# branched is excluded: its tip frame (tipA) is independent of j3, so the tip
# Jacobian carries a permanent zero column → it is *always* rank-deficient, which
# has no well-conditioned samples to cross-check κ against. The singular case is
# covered by the golden tests below; branched's Jacobian itself is already
# cross-validated against Pinocchio in test_jacobian_matches_pinocchio.
ANALYSIS_FIXTURES = [f for f in FIXTURES if f != "branched"] + ["showcase6"]


@pytest.mark.skipif(
    not _have_analyze(),
    reason="caliper.Robot lacks analyze/manipulability -- rebuild bindings after Phase-2",
)
@pytest.mark.parametrize("name", ANALYSIS_FIXTURES)
def test_analysis_matches_numpy_svd(name):
    """analyze(q).{manipulability,condition_number,sigma_min,sigma} == numpy SVD of
    the Pinocchio LOCAL_WORLD_ALIGNED Jacobian. Caliper analyze uses World==LWA;
    geometric-Jacobian singular values are frame-invariant, so they must match."""
    model, data, robot = _load(name)
    assert model.nq == robot.ndof and model.nv == robot.ndof, (
        f"[{name}] multi-DoF joint broke the idx mapping"
    )
    fid = model.getFrameId(robot.tip_frame())
    perm = _column_perm_pin_to_caliper(model, robot)
    rng = _rng(name)
    ran = 0
    saw_well_conditioned = False
    for s in range(N_SAMPLES):
        q = _sample_q(rng, model, robot)
        q_pin = _q_to_pinocchio(model, robot, q)
        pin.computeJointJacobians(model, data, q_pin)
        pin.updateFramePlacements(model, data)
        J = np.asarray(
            pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        )[:, perm]
        sig = np.linalg.svd(J, compute_uv=False)  # descending
        manip_ref = float(sig.prod())
        smin_ref = float(sig.min())
        cond_ref = float(sig.max() / smin_ref) if smin_ref > 0.0 else math.inf

        rep = robot.analyze(list(q))
        assert math.isclose(rep["sigma_min"], smin_ref, rel_tol=1e-7, abs_tol=1e-9), (
            f"[{name}] s={s} sigma_min {rep['sigma_min']:.6e} vs numpy {smin_ref:.6e}\nq={q.tolist()}"
        )
        assert math.isclose(rep["manipulability"], manip_ref, rel_tol=1e-6, abs_tol=1e-12), (
            f"[{name}] s={s} manip {rep['manipulability']:.6e} vs numpy {manip_ref:.6e}\nq={q.tolist()}"
        )
        ec = rep["condition_number"]
        if math.isfinite(cond_ref) and cond_ref < 1e6:
            saw_well_conditioned = True
            assert math.isclose(ec, cond_ref, rel_tol=1e-6), (
                f"[{name}] s={s} cond {ec:.6e} vs numpy {cond_ref:.6e}\nq={q.tolist()}"
            )
        else:
            # ill-conditioned / structurally singular: numpy may report ∞ while a
            # finite-precision SVD reports a huge finite κ — agreement = "both large".
            assert ec is None or ec > 1e6, (
                f"[{name}] s={s} ill-cond engine κ={ec} not large (numpy {cond_ref:.3e})"
            )
        # engine pads `sigma` to length 3 with trailing zeros (k<3 robots); match it.
        want_sigma = np.zeros(3)
        k = min(3, sig.shape[0])
        want_sigma[:k] = np.sort(sig)[:k]
        np.testing.assert_allclose(
            rep["sigma"], want_sigma, rtol=1e-7, atol=1e-9,
            err_msg=f"[{name}] s={s} sigma-triple mismatch",
        )
        assert math.isclose(
            robot.manipulability(list(q)), manip_ref, rel_tol=1e-6, abs_tol=1e-12
        )
        ran += 1
    assert ran == N_SAMPLES, f"[{name}] analysis sweep ran {ran}/{N_SAMPLES}"
    assert saw_well_conditioned, (
        f"[{name}] EVERY sample ill-conditioned -- tight cond cross-check never ran"
    )


# tag, q, robust-offending-anchor (subset that MUST be present), allowed-superset
GOLDEN_SINGULAR = [
    ("wrist", [0.3, 0.5, -0.7, 0.4, 0.0, -0.2], {3, 5}, {3, 5}),  # exact: robust
    ("elbow", [0.2, 0.4, 0.0, 0.3, 0.7, 0.1], {2}, {1, 2}),  # j3 robust; j2 knife-edge
]


@pytest.mark.skipif(
    not _have_analyze(),
    reason="caliper.Robot lacks analyze/manipulability -- rebuild bindings after Phase-2",
)
@pytest.mark.parametrize("tag,q,must_have,allowed", GOLDEN_SINGULAR)
def test_golden_singular_showcase6(tag, q, must_have, allowed):
    model, data, robot = _load("showcase6")
    assert robot.joint_names == ["j1", "j2", "j3", "j4", "j5", "j6"], (
        f"showcase6 joint order changed ({robot.joint_names}); golden configs assume j1..j6"
    )
    # Independent witness: Pinocchio's own LWA Jacobian is singular here too.
    fid = model.getFrameId(robot.tip_frame())
    perm = _column_perm_pin_to_caliper(model, robot)
    pin.computeJointJacobians(model, data, _q_to_pinocchio(model, robot, np.array(q)))
    pin.updateFramePlacements(model, data)
    J = np.asarray(
        pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    )[:, perm]
    s_ref = np.linalg.svd(J, compute_uv=False)
    assert s_ref.min() < 1e-6, f"[{tag}] not singular in Pinocchio (smin={s_ref.min():.2e})"

    rep = robot.analyze(list(q))
    assert rep["sigma_min"] < 1e-6, f"[{tag}] engine sigma_min={rep['sigma_min']:.2e} not singular"
    assert rep["manipulability"] < 1e-9, f"[{tag}] manip={rep['manipulability']:.2e} not ~0"
    assert rep["condition_number"] > 1e9, f"[{tag}] cond={rep['condition_number']:.2e} not large"
    assert rep["kind"] != "none", f"[{tag}] kind=none at a singular config"
    off = set(rep["offending_joints"])
    # robust joint-space contract: anchor pair MUST be present, no foreign joints leak
    assert must_have <= off, f"[{tag}] offending {off} missing anchor {must_have}"
    assert off <= allowed, f"[{tag}] offending {off} leaked outside {allowed}"
    # numpy and engine agree the config is (equally) singular
    assert math.isclose(rep["sigma_min"], float(s_ref.min()), rel_tol=1e-6, abs_tol=1e-9)


@pytest.mark.skipif(not _have_analyze(), reason="needs Phase-2 analyze binding")
def test_showcase6_generic_well_conditioned():
    _, _, robot = _load("showcase6")
    rep = robot.analyze([0.3, -0.4, 0.6, 0.2, -0.5, 0.1])
    assert rep["kind"] == "none"
    assert rep["sigma_min"] > 1e-2
    assert rep["condition_number"] < 1e3
    assert rep["manipulability"] > 1e-5
