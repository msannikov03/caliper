"""Real-robot corpus oracle: cross-validate Caliper FK / Jacobians against
Pinocchio on REAL vendored URDFs (``oracle/fixtures/corpus/``), killing the
single-fixture bias of the hand-authored ``robots/`` fixtures.

Corpus (vendored verbatim; see ``oracle/fixtures/corpus/README.md``):
  * ``panda.urdf``            -- Franka Panda, 9 dof incl. a mimic finger,
                                 ``package://`` mesh URIs
  * ``so101_new_calib.urdf``  -- SO-101, 6 dof, relative ``assets/*.stl``
  * ``so100.urdf``            -- SO-100, 6 dof
  * ``gen3_lite.urdf``        -- Kinova Gen3 lite, 10 dof incl. 3 mimic finger
                                 joints, ``package://`` mesh URIs

Meshes are deliberately NOT vendored (~58 MB): visuals are infallible-by-design
(kept with ``path=None``) and unloadable colliders are dropped LOUDLY via
``dropped_collider_frames`` -- neither affects the kinematics under test.

Conventions follow ``test_pinocchio.py`` exactly (verified there against the
source): [v; w] twist ordering on both sides, Caliper ``jacobian(..., "world")``
== Pinocchio ``LOCAL_WORLD_ALIGNED``, q scattered by joint NAME via ``idx_q``,
Jacobian columns reordered via ``idx_v``.

Mimic note: panda and gen3_lite carry ``<mimic>`` joints. Pinocchio 4's DEFAULT
build (``mimic=False``) keeps them as independent dofs -- which matches
Caliper's FULL-SPACE model (mimics stay in the arrays) -- so the cross-check
runs in FULL space here. Reduced-space mimic parity is already covered by
``test_mimic.py``.
"""
import hashlib
import pathlib
import time

import numpy as np
import pytest

caliper = pytest.importorskip("caliper", reason="caliper bindings not built")
pin = pytest.importorskip("pinocchio", reason="pinocchio (pin) not installed")

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = ROOT / "oracle" / "fixtures" / "corpus"

# name -> expected ndof (mimics counted: Caliper is full-space, pin-4 default too)
ROBOTS = {
    "panda": 9,
    "so101_new_calib": 6,
    "so100": 6,
    "gen3_lite": 10,
}

FK_ATOL = 1e-10
JAC_ATOL = 1e-9
N_SAMPLES = 5
SEED = 0xC0B5  # deterministic, corpus-wide

# ~/.cache source of the vendored corpus; only the cache has the real meshes.
CACHE = pathlib.Path.home() / ".cache" / "robot_descriptions"
# (relative cache path, expected ndof) for the real-mesh integration check
REAL_MESH_ROBOTS = [
    ("SO-ARM100/Simulation/SO101/so101_new_calib.urdf", 6),
    ("example-robot-data/robots/panda_description/urdf/panda.urdf", 9),
]
LOAD_BOUND_S = 60.0  # R1.5a hull bound: real meshes must load in < 60s (debug)


def _urdf(name):
    return str(CORPUS / f"{name}.urdf")


def _load(name):
    """Build the Pinocchio (fixed-base, full-space) model+data and the Caliper
    robot from the SAME vendored file."""
    path = _urdf(name)
    model = pin.buildModelFromUrdf(path)
    data = model.createData()
    robot = caliper.Robot.from_urdf(path)
    return model, data, robot


def _rng(name):
    # Per-robot stream, salt-independent (stable digest) -- reproducible alone.
    tag = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return np.random.default_rng([SEED, tag])


def _guard_scalar_mapping(name, model, robot):
    """The by-name idx_q/idx_v scatter assumes one config coord per joint."""
    assert model.nq == robot.ndof, (
        f"[{name}] nq ({model.nq}) != ndof ({robot.ndof}) -- a multi-DoF joint "
        "(e.g. 'continuous') broke the 1:1 idx_q mapping"
    )
    assert model.nv == robot.ndof, f"[{name}] nv ({model.nv}) != ndof ({robot.ndof})"


def _sample_q(rng, model, robot):
    """A random config INSIDE the real joint limits, in Caliper joint order.
    A continuous/unbounded joint (pin reports +/-inf) samples [-pi, pi]."""
    q = np.empty(robot.ndof)
    for i, jname in enumerate(robot.joint_names):
        iq = model.joints[model.getJointId(jname)].idx_q
        lo = model.lowerPositionLimit[iq]
        hi = model.upperPositionLimit[iq]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = -np.pi, np.pi
        q[i] = rng.uniform(lo, hi)
    return q


def _q_to_pinocchio(model, robot, q):
    q_pin = np.zeros(model.nq)
    for i, jname in enumerate(robot.joint_names):
        q_pin[model.joints[model.getJointId(jname)].idx_q] = q[i]
    return q_pin


def _column_perm(model, robot):
    return [model.joints[model.getJointId(n)].idx_v for n in robot.joint_names]


def _frames_to_check(model, robot):
    """The tip frame plus ~3 intermediate link frames both sides expose,
    deterministically spread along the frame list."""
    tip = robot.tip_frame()
    assert model.existFrame(tip), f"tip frame '{tip}' not in Pinocchio model"
    shared = [f for f in robot.frame_names() if f != tip and model.existFrame(f)]
    assert shared, "no shared intermediate frames -- comparison would be tip-only"
    stride = max(1, len(shared) // 3)
    return [tip] + shared[::stride][:3]


@pytest.mark.parametrize("name,ndof", sorted(ROBOTS.items()))
def test_corpus_compiles(name, ndof):
    robot = caliper.Robot.from_urdf(_urdf(name))
    assert robot.ndof == ndof, (
        f"[{name}] ndof {robot.ndof} != expected {ndof} (full-space, mimics counted)"
    )


@pytest.mark.parametrize("name", sorted(ROBOTS))
def test_corpus_fk_vs_pinocchio(name):
    model, data, robot = _load(name)
    _guard_scalar_mapping(name, model, robot)
    frames = _frames_to_check(model, robot)

    rng = _rng(name)
    compared = 0
    moved = False
    for s in range(N_SAMPLES):
        q = _sample_q(rng, model, robot)
        pin.forwardKinematics(model, data, _q_to_pinocchio(model, robot, q))
        pin.updateFramePlacements(model, data)
        for fname in frames:
            T_pin = np.asarray(data.oMf[model.getFrameId(fname)].homogeneous)
            T_cal = np.asarray(robot.fk(q, fname))
            assert T_cal.shape == (4, 4)
            compared += 1
            if not np.allclose(T_cal, np.eye(4), atol=1e-6):
                moved = True
            assert np.max(np.abs(T_cal - T_pin)) <= FK_ATOL, (
                f"FK mismatch [{name}] frame='{fname}' sample={s}: "
                f"worst |delta|={np.max(np.abs(T_cal - T_pin)):.3e} "
                f"(atol={FK_ATOL})\nq={q.tolist()}\n"
                f"caliper=\n{T_cal}\npinocchio=\n{T_pin}"
            )
    assert compared == N_SAMPLES * len(frames), f"[{name}] FK sweep was cut short"
    assert moved, f"[{name}] every compared frame was identity -- vacuous sweep"


@pytest.mark.parametrize("name", sorted(ROBOTS))
def test_corpus_jacobian_vs_pinocchio(name):
    model, data, robot = _load(name)
    _guard_scalar_mapping(name, model, robot)
    tip = robot.tip_frame()
    fid = model.getFrameId(tip)
    perm = _column_perm(model, robot)

    rng = _rng(name)
    ran = 0
    for s in range(N_SAMPLES):
        q = _sample_q(rng, model, robot)
        q_pin = _q_to_pinocchio(model, robot, q)
        pin.forwardKinematics(model, data, q_pin)
        pin.computeJointJacobians(model, data, q_pin)
        pin.updateFramePlacements(model, data)

        J_pin = np.asarray(
            pin.getFrameJacobian(
                model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
        )[:, perm]
        J_cal = np.asarray(robot.jacobian(q, tip, "world"))
        assert J_cal.shape == J_pin.shape
        assert np.max(np.abs(J_cal - J_pin)) <= JAC_ATOL, (
            f"Jacobian(world) mismatch [{name}] frame='{tip}' sample={s}: "
            f"worst |delta|={np.max(np.abs(J_cal - J_pin)):.3e} "
            f"(atol={JAC_ATOL})\nq={q.tolist()}\n"
            f"caliper=\n{J_cal}\npinocchio=\n{J_pin}"
        )
        ran += 1
    assert ran == N_SAMPLES, f"[{name}] Jacobian sweep ran {ran}/{N_SAMPLES}"


@pytest.mark.skipif(
    not CACHE.is_dir(),
    reason="~/.cache/robot_descriptions absent -- real meshes unavailable (CI)",
)
@pytest.mark.parametrize("rel,ndof", REAL_MESH_ROBOTS)
def test_corpus_real_meshes_bounded(rel, ndof):
    """Integration (dev machines only): loading the REAL URDF from the cache --
    collision meshes resolving and hulling for real -- stays under the R1.5a
    bound, and the model is sane (correct ndof, finite FK at q=0)."""
    path = CACHE / rel
    if not path.is_file():
        pytest.skip(f"{rel} not in the local robot_descriptions cache")
    t0 = time.monotonic()
    robot = caliper.Robot.from_urdf(str(path))
    elapsed = time.monotonic() - t0
    assert elapsed < LOAD_BOUND_S, (
        f"{rel} took {elapsed:.1f}s to load -- R1.5a hull bound regressed"
    )
    assert robot.ndof == ndof
    T = np.asarray(robot.fk(np.zeros(ndof), robot.tip_frame()))
    assert T.shape == (4, 4) and np.all(np.isfinite(T)), "FK at q=0 not finite"
