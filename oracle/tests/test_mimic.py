"""Mimic-joint oracle: cross-validate Caliper's mimic handling against Pinocchio.

Caliper's architectural choice (deliberate, additive): mimic joints STAY in the
full-space arrays -- ``Robot.ndof`` / ``fk(q, ...)`` remain FULL-SPACE -- and the
Model merely records the constraints, with reduced<->full mapping helpers plus
reduced-space FK/Jacobian wrappers in ``caliper-kinematics``. Pinocchio 4 can
instead build the REDUCED model directly (``buildModelFromUrdf(path, mimic=True)``
-- verified: nq drops by one per mimic). So the cross-check is:

    pin (reduced, mimic=True) FK at q_red
        ==  Caliper full-space FK at q_full = expand(q_red)

where the expansion ``q[i] = multiplier*q[source] + offset`` is computed HERE in
python from the fixture's known constants (wrist = 0.5*arm + 0.1, finger2 =
-finger1). The PyO3 face does not yet expose ``expand_mimic`` or the reduced
wrappers (face wiring is a later step) -- when it does, the expansion below
should switch to ``robot.expand_mimic`` and this comment be updated.

If this Pinocchio build lacks the ``mimic=True`` kwarg, the reduced cross-check
SKIPs with a clear reason and the fallback test still cross-validates the same
math: pin's FULL model (both fingers independent) fed the expanded configuration
must match Caliper's full-space FK -- an equivalent check of the expanded-q path.
"""
import pathlib

import numpy as np
import pytest

caliper = pytest.importorskip("caliper", reason="caliper bindings not built")
pin = pytest.importorskip("pinocchio", reason="pinocchio (pin) not installed")

ROOT = pathlib.Path(__file__).resolve().parents[2]
URDF = str(ROOT / "oracle" / "fixtures" / "robots" / "gripper_mimic.urdf")

ATOL = 1e-9
N_SAMPLES = 25
SEED = 0x313C  # deterministic

# Fixture ground truth (must match gripper_mimic.urdf exactly).
# Caliper full-space joint order (topological): arm, wrist, finger1, finger2.
FULL_JOINTS = ["arm", "wrist", "finger1", "finger2"]
INDEPENDENT = ["arm", "finger1"]
# name -> (source name, multiplier, offset)
MIMICS = {"wrist": ("arm", 0.5, 0.1), "finger2": ("finger1", -1.0, 0.0)}
FRAMES = ["upper_arm", "palm", "finger_l", "finger_r"]


def _expand(q_red):
    """q_full (Caliper joint order) from independent [arm, finger1] values."""
    by_name = dict(zip(INDEPENDENT, q_red))
    full = []
    for name in FULL_JOINTS:
        if name in MIMICS:
            src, mult, off = MIMICS[name]
            full.append(mult * by_name[src] + off)
        else:
            full.append(by_name[name])
    return np.array(full)


def _pin_mimic_model():
    """The mimic-REDUCED Pinocchio model, or None if unsupported by this build."""
    try:
        model = pin.buildModelFromUrdf(URDF, mimic=True)
    except TypeError:
        return None
    if model.nq != len(INDEPENDENT):  # kwarg accepted but not actually reducing
        return None
    return model


def _sample_reduced(rng):
    # inside the URDF limits of arm [-2.9, 2.9] (keep wrist=0.5q+0.1 in [-1.6, 1.6])
    # and finger1 [0, 0.04]
    return np.array([rng.uniform(-2.9, 2.9), rng.uniform(0.0, 0.04)])


def _caliper_robot():
    robot = caliper.Robot.from_urdf(URDF)
    assert robot.ndof == 4, "Caliper keeps mimics as full-space dofs"
    assert list(robot.joint_names) == FULL_JOINTS
    return robot


def test_reduced_pin_vs_caliper_expanded_fk():
    """pin's mimic-reduced FK == Caliper's full-space FK on the expanded config."""
    model = _pin_mimic_model()
    if model is None:
        pytest.skip(
            "this pinocchio build does not support buildModelFromUrdf(..., mimic=True)"
        )
    data = model.createData()
    robot = _caliper_robot()

    # map reduced q by joint name into pin's reduced config vector
    idx_q = {name: model.joints[model.getJointId(name)].idx_q for name in INDEPENDENT}

    rng = np.random.default_rng(SEED)
    compared = 0
    for _ in range(N_SAMPLES):
        q_red = _sample_reduced(rng)
        q_pin = np.zeros(model.nq)
        for name, v in zip(INDEPENDENT, q_red):
            q_pin[idx_q[name]] = v
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)

        q_full = _expand(q_red)
        for fname in FRAMES:
            assert model.existFrame(fname), f"pin lacks frame '{fname}'"
            T_pin = np.asarray(data.oMf[model.getFrameId(fname)].homogeneous)
            T_cal = np.asarray(robot.fk(q_full, fname))
            compared += 1
            assert np.max(np.abs(T_cal - T_pin)) <= ATOL, (
                f"reduced-mimic FK mismatch at frame '{fname}': "
                f"q_red={q_red.tolist()} q_full={q_full.tolist()}\n"
                f"caliper=\n{T_cal}\npinocchio(reduced)=\n{T_pin}"
            )
    assert compared == N_SAMPLES * len(FRAMES)

    # non-vacuity: the mimics genuinely move their links (finger_r depends on
    # finger1 THROUGH the mimic; wrist pose depends on arm through m=0.5, b=0.1)
    Ta = np.asarray(robot.fk(_expand([0.3, 0.01]), "finger_r"))
    Tb = np.asarray(robot.fk(_expand([0.3, 0.03]), "finger_r"))
    assert not np.allclose(Ta, Tb, atol=1e-6), "mimicked finger must move"


def test_full_space_pin_vs_caliper_expanded_fk():
    """Equivalent full-space check (runs regardless of pin mimic support): pin's
    FULL model -- both fingers independent -- fed the SAME expanded configuration
    must agree with Caliper's full-space FK."""
    model = pin.buildModelFromUrdf(URDF)  # mimics kept as ordinary joints
    assert model.nq == 4
    data = model.createData()
    robot = _caliper_robot()

    idx_q = {name: model.joints[model.getJointId(name)].idx_q for name in FULL_JOINTS}

    rng = np.random.default_rng(SEED + 1)
    for _ in range(N_SAMPLES):
        q_full = _expand(_sample_reduced(rng))
        q_pin = np.zeros(model.nq)
        for name, v in zip(FULL_JOINTS, q_full):
            q_pin[idx_q[name]] = v
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)
        for fname in FRAMES:
            T_pin = np.asarray(data.oMf[model.getFrameId(fname)].homogeneous)
            T_cal = np.asarray(robot.fk(q_full, fname))
            assert np.max(np.abs(T_cal - T_pin)) <= ATOL, (
                f"full-space FK mismatch at frame '{fname}': q={q_full.tolist()}\n"
                f"caliper=\n{T_cal}\npinocchio(full)=\n{T_pin}"
            )
