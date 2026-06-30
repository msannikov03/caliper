"""External oracle for the SE(3) log map and the IK SOLVER.

Closes two specific "self-consistent only" gaps the rest of the suite leaves open:

  1. ``caliper.log6`` / ``caliper.exp6`` were only ever round-tripped against each
     other (and, in test_pinocchio's commentary, deliberately NOT compared to an
     outside reference). Here we pin ``log6`` to ``scipy.linalg.logm`` — an
     independent matrix-logarithm code path — element-wise, then confirm
     ``exp6(log6(T)) == T``.

  2. The IK solver was only validated by FK-closure (does ``fk(ik(target))`` hit
     ``target``?). That cannot distinguish a correct solver from one that merely
     drives the residual it happens to measure. Here we re-implement IK from
     scratch in pure NumPy as a *task-space* damped-least-squares loop
     (``dq = Jᵀ(JJᵀ + λ²I)⁻¹ e``) — a genuinely different formulation from the
     engine's joint-space Levenberg–Marquardt (``(JᵀJ + …)dq = Jᵀe``) — using only
     ``caliper.jacobian`` (already Pinocchio-cross-validated) and ``caliper.log6``
     for the error twist. We then assert the engine's ``Robot.ik`` reaches the
     SAME tip pose as this independent solver. Matching an outside DLS, not just
     its own FK, is the load-bearing signal.

Conventions (read from the source, not assumed):
  * Twists are ``[v; ω]`` (linear first) everywhere — ``Twist``, the Jacobian rows,
    ``log6``/``exp6``. No row swap needed; the only care point is mapping scipy's
    se(3) matrix ``[[skew(ω), v],[0,0]]`` onto ``[v; ω]``.
  * ``log6(m)`` / ``exp6(twist)`` use ROW-MAJOR 4×4 (``m[row][col]``), same as
    ``Robot.fk``. ``Robot.ik`` takes a COLUMN-MAJOR target (``target[col][row]``),
    so we transpose fk's row-major pose before feeding it in.
  * Both the engine IK and our NumPy DLS use the body-frame error twist
    ``e = log6(T_cur⁻¹ · T_target)`` with the BODY ("LOCAL") Jacobian — required
    for the two to converge to the same manifold; the DLS *step rule* is what
    differs.

Skipped wholesale when caliper / scipy are absent, and per-module when the binding
predates ``log6`` / ``exp6`` / ``Robot.ik`` (rebuild with ``maturin develop``).
"""
import hashlib
import pathlib

import numpy as np
import pytest

import caliper

splinalg = pytest.importorskip("scipy.linalg", reason="scipy not installed")

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"

SEED = 0x106F6  # deterministic, fixture-independent

# ---- log6 / exp6 sweep ------------------------------------------------------
N_LOG = 200
ANGLE_LO, ANGLE_HI = 0.05, 2.8  # bounded away from π so logm == engine principal log
LOG_ATOL = 1e-8  # scipy.linalg.logm is ~1e-12 accurate here; this is generous slack
RT_ATOL = 1e-9  # exp6(log6(T)) round-trip

# ---- IK cross-check ---------------------------------------------------------
# >=6 DoF fixtures only: a general SE(3) target needs a full-rank task Jacobian for
# the independent task-space DLS to converge cleanly. (toy=2DoF, prismatic=1DoF,
# branched has a structurally rank-deficient tip — all excluded.)
IK_FIXTURES = ["showcase6", "redundant7"]
N_IK = 12
IK_DAMP = 1e-4  # small Tikhonov damping: ~Gauss–Newton away from singularities
IK_MAX_ITERS = 400
IK_TOL = 1e-10  # DLS convergence target (||e||)
DLS_RES_TOL = 1e-6  # the independent solver must actually converge this far
POSE_ATOL = 1e-5  # engine pose vs DLS pose vs target


def _have_log6():
    return hasattr(caliper, "log6") and hasattr(caliper, "exp6")


def _have_ik():
    return hasattr(caliper.Robot, "ik") and hasattr(caliper.Robot, "jacobian")


def _rng(tag):
    """Stable per-stream RNG (sha256 digest, not the salt-randomized builtin hash)
    so the sweep is byte-for-byte reproducible across processes."""
    t = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "little")
    return np.random.default_rng([SEED, t])


def _rodrigues(axis, angle):
    """Rotation matrix from axis-angle (independent of scipy, for generating poses)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _random_se3(rng):
    """A random homogeneous 4×4 with rotation angle in [ANGLE_LO, ANGLE_HI]."""
    axis = rng.normal(size=3)
    angle = rng.uniform(ANGLE_LO, ANGLE_HI)
    t = rng.uniform(-1.0, 1.0, size=3)
    T = np.eye(4)
    T[:3, :3] = _rodrigues(axis, angle)
    T[:3, 3] = t
    return T


def _twist_from_logm(T):
    """scipy matrix log of a 4×4 SE(3) → twist [v; ω].

    ``logm(T)`` is the se(3) matrix ``[[skew(ω), v],[0,0]]``; v is its top-right
    column, ω is the vee of the skew block. We assemble [v; ω] to match the engine.
    """
    M = np.real(splinalg.logm(T))
    v = M[:3, 3]
    omega = np.array([M[2, 1], M[0, 2], M[1, 0]])
    return np.concatenate([v, omega])


@pytest.mark.skipif(
    not _have_log6(),
    reason="caliper lacks log6/exp6 — rebuild bindings (maturin develop)",
)
def test_log6_matches_scipy_logm():
    rng = _rng("log6")
    max_err = 0.0
    saw_rotation = False
    for s in range(N_LOG):
        T = _random_se3(rng)
        twist_cal = np.asarray(caliper.log6(T.tolist()))
        assert twist_cal.shape == (6,), f"log6 returned shape {twist_cal.shape}, want (6,)"
        twist_ref = _twist_from_logm(T)
        err = float(np.max(np.abs(twist_cal - twist_ref)))
        max_err = max(max_err, err)
        if np.linalg.norm(twist_ref[3:]) > 1e-3:
            saw_rotation = True
        np.testing.assert_allclose(
            twist_cal, twist_ref, atol=LOG_ATOL, rtol=0.0,
            err_msg=(
                f"log6 vs scipy.logm mismatch (sample {s}): worst |Δ|={err:.3e}\n"
                f"caliper={twist_cal}\nscipy  ={twist_ref}\nT=\n{T}"
            ),
        )
    assert saw_rotation, "log6 sweep generated no real rotation — vacuous"
    assert max_err <= LOG_ATOL  # explicit non-vacuity on the bound itself


@pytest.mark.skipif(
    not _have_log6(),
    reason="caliper lacks log6/exp6 — rebuild bindings (maturin develop)",
)
def test_exp6_log6_roundtrip():
    """exp6(log6(T)) == T element-wise (closes the exp/log loop on the binding)."""
    rng = _rng("exp6")
    for s in range(N_LOG):
        T = _random_se3(rng)
        twist = caliper.log6(T.tolist())
        T_rt = np.asarray(caliper.exp6(twist))
        assert T_rt.shape == (4, 4), f"exp6 returned shape {T_rt.shape}, want (4,4)"
        np.testing.assert_allclose(
            T_rt, T, atol=RT_ATOL, rtol=0.0,
            err_msg=f"exp6(log6(T)) != T (sample {s})\nT=\n{T}\nrt=\n{T_rt}",
        )
        # And exp6 of scipy's independently-computed twist also reconstructs T:
        T_ref = np.asarray(caliper.exp6(_twist_from_logm(T).tolist()))
        np.testing.assert_allclose(
            T_ref, T, atol=1e-7, rtol=0.0,
            err_msg=f"exp6(scipy twist) != T (sample {s})\nT=\n{T}\nrt=\n{T_ref}",
        )


# ===== IK SOLVER vs independent NumPy task-space DLS =====


def _load(name):
    return caliper.Robot.from_urdf(str(ROBOTS / f"{name}.urdf"))


def _sample_q(rng, robot):
    """Random config inside joint limits (clamped to [-2, 2]; [-2, 2] when a limit
    is missing/infinite). Returned in robot.joint_names order."""
    q = np.empty(robot.ndof)
    for i, lim in enumerate(robot.joint_limits):
        if lim is None:
            lo, hi = -2.0, 2.0
        else:
            lo, hi = lim
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                lo, hi = -2.0, 2.0
            else:
                lo, hi = max(lo, -2.0), min(hi, 2.0)
                if hi <= lo:
                    lo, hi = -2.0, 2.0
        q[i] = rng.uniform(lo, hi)
    return q


def _clamp_to_limits(robot, q):
    out = np.array(q, dtype=float)
    for i, lim in enumerate(robot.joint_limits):
        if lim is not None:
            lo, hi = lim
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                out[i] = min(max(out[i], lo), hi)
    return out


def _numpy_dls_ik(robot, tip, target_T, seed):
    """Independent task-space damped-least-squares IK, pure NumPy.

    Uses ONLY caliper.jacobian (body frame) + caliper.log6 (body error twist).
    Step rule ``dq = Jᵀ(JJᵀ + λ²I)⁻¹ e`` is deliberately the task-space DLS, not
    the engine's joint-space LM. Returns (q, residual, iters).
    """
    q = np.array(seed, dtype=float)
    err = np.inf
    for it in range(IK_MAX_ITERS):
        T_cur = np.asarray(robot.fk(list(q), tip))
        T_err = np.linalg.inv(T_cur) @ target_T
        e = np.asarray(caliper.log6(T_err.tolist()))  # body twist [v; ω]
        err = float(np.linalg.norm(e))
        if err < IK_TOL:
            return q, err, it
        J = np.asarray(robot.jacobian(list(q), tip, "body"))  # 6 × n, rows [v; ω]
        dq = J.T @ np.linalg.solve(J @ J.T + (IK_DAMP ** 2) * np.eye(6), e)
        # trust-region clamp on the step to keep the loop stable far from solution
        nrm = np.linalg.norm(dq)
        if nrm > 0.5:
            dq *= 0.5 / nrm
        q = q + dq
    return q, err, IK_MAX_ITERS


def _fk_target_colmajor(robot, tip, q):
    """fk() row-major pose → column-major nested list for Robot.ik (target[col][row])."""
    T = np.asarray(robot.fk(list(q), tip))
    return T, T.T.tolist()


@pytest.mark.skipif(
    not (_have_ik() and _have_log6()),
    reason="caliper lacks ik/jacobian/log6 — rebuild bindings (maturin develop)",
)
@pytest.mark.parametrize("name", IK_FIXTURES)
def test_ik_solver_matches_numpy_dls(name):
    robot = _load(name)
    assert robot.ndof >= 6, f"[{name}] ndof={robot.ndof} < 6 — target may be unreachable"
    tip = robot.tip_frame()
    rng = _rng(f"ik:{name}")
    ran = 0
    for s in range(N_IK):
        # Reachable target by construction: FK of an in-limit config.
        q_target = _sample_q(rng, robot)
        target_T, target_cm = _fk_target_colmajor(robot, tip, q_target)

        # Seed near (but distinct from) the solution so the independent DLS has a
        # good basin — the point is to validate solver MATH, not basin size.
        seed = _clamp_to_limits(robot, q_target + rng.uniform(-0.25, 0.25, robot.ndof))

        # --- independent NumPy task-space DLS ---
        q_dls, res_dls, _ = _numpy_dls_ik(robot, tip, target_T, seed)
        assert res_dls < DLS_RES_TOL, (
            f"[{name}] s={s}: independent NumPy DLS failed to converge "
            f"(residual={res_dls:.3e}); seed/target may be near a singularity"
        )

        # --- engine IK ---
        res = robot.ik(target_cm, list(seed))
        assert res["success"], (
            f"[{name}] s={s}: engine IK reported failure (residual={res['residual']:.3e})"
        )
        assert res["residual"] < 1e-6, (
            f"[{name}] s={s}: engine IK residual {res['residual']:.3e} not small"
        )
        q_ik = np.asarray(res["q"])

        # --- both must reach the SAME tip pose (compare FK, not q: redundant arms
        #     and 6-DoF arms alike admit multiple joint solutions to one pose) ---
        T_dls = np.asarray(robot.fk(list(q_dls), tip))
        T_ik = np.asarray(robot.fk(list(q_ik), tip))
        np.testing.assert_allclose(
            T_dls, target_T, atol=POSE_ATOL, rtol=0.0,
            err_msg=f"[{name}] s={s}: NumPy DLS pose != target",
        )
        np.testing.assert_allclose(
            T_ik, target_T, atol=POSE_ATOL, rtol=0.0,
            err_msg=f"[{name}] s={s}: engine IK pose != target",
        )
        np.testing.assert_allclose(
            T_ik, T_dls, atol=POSE_ATOL, rtol=0.0,
            err_msg=(
                f"[{name}] s={s}: engine IK and independent DLS reached DIFFERENT "
                f"poses\nengine=\n{T_ik}\nnumpy-dls=\n{T_dls}"
            ),
        )
        ran += 1
    assert ran == N_IK, f"[{name}] IK sweep ran {ran}/{N_IK} — non-vacuity guard tripped"
