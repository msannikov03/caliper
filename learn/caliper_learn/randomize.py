"""Domain randomization for the sim substrate: seeded, diffable draws over
physics/camera/spawn parameters, applied either to the MJCF document (model
compile-time params) or to a live `VecSimEnv` (runtime params).

The split matters and is deliberate:

- MODEL-level params (body mass, joint damping/frictionloss, gravity) are
  baked into the compiled `mujoco.MjModel`, so randomizing them means editing
  the MJCF **string** and recompiling — `apply_to_mjcf` is a deterministic
  XML edit (tested by parse-compare, never regex), and any env using it must
  own a PER-ENV model copy (see the VecSimEnv doc for the honest memory cost).
- ENV-level params (PD gains, camera pose, spawn offset) live outside the
  compiled model and are applied in place by `apply_to_env` — no recompile.

A draw is a PLAIN dict of python floats/lists — json-serializable and
line-diffable, so a CI job can `json.dumps` two draws and diff them (the
reproducibility contract: log the draw, replay the run). `sample` is
splitmix-style seeded: same spec + same seed → the exact-equal draw, and the
field order is FIXED (`_FIELDS`) so a spec that randomizes fewer fields does
not shift the stream of the ones it keeps.

Ranges are `(low, high)` tuples; whether they are MULTIPLIERS or ABSOLUTE
jitter is per field and stated on the field — multipliers for quantities with
a meaningful baseline (mass, gains), absolute for quantities whose baseline
is zero or arbitrary (damping, friction, offsets).

Heavy deps (mujoco) are imported lazily inside `apply_to_env`, matching the
package rule; `sample`/`apply_to_mjcf` need only numpy + stdlib.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

Range = tuple[float, float]

# Draw keys that require an MJCF edit + model recompile.
MODEL_KEYS = frozenset({"mass_scale", "joint_damping", "joint_frictionloss", "gravity"})
# Draw keys applied to a live env, no recompile.
ENV_KEYS = frozenset({"kp_scale", "kd_scale", "camera_pos", "camera_euler", "spawn_offset"})

# (spec field, draw key, per-draw length: "dof" | "xyz" | "scalar")
_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("mass_scale", "mass_scale", "dof"),
    ("joint_damping", "joint_damping", "dof"),
    ("joint_frictionloss", "joint_frictionloss", "dof"),
    ("kp_scale", "kp_scale", "scalar"),
    ("kd_scale", "kd_scale", "scalar"),
    ("camera_pos_jitter", "camera_pos", "xyz"),
    ("camera_euler_jitter", "camera_euler", "xyz"),
    ("spawn_jitter", "spawn_offset", "dof"),
    ("gravity_jitter", "gravity", "xyz"),
)

# Fields whose values multiply a positive baseline: low must stay > 0.
_MULTIPLIERS = frozenset({"mass_scale", "kp_scale", "kd_scale"})


@dataclass(frozen=True)
class RandomizationSpec:
    """What to randomize and how much. `None` = leave that field alone.

    MULTIPLIER ranges (drawn factor × baseline, low must be > 0):
      `mass_scale`   — per BODY, scales `<inertial>` mass AND its inertia
                       (fixed geometry: inertia is proportional to mass);
      `kp_scale` / `kd_scale` — per env, scales the computed-torque PD gains.

    ABSOLUTE ranges (drawn value used directly / added to the baseline):
      `joint_damping`       — per joint, N·m·s/rad, SETS `<joint damping>`;
      `joint_frictionloss`  — per joint, N·m (dry friction), SETS
                              `<joint frictionloss>`;
      `camera_pos_jitter`   — per axis, meters, added to the camera position;
      `camera_euler_jitter` — per axis, radians, XYZ rotation composed onto
                              the camera orientation;
      `spawn_jitter`        — per dof, added to the reset qpos (clipped to
                              the joint limits);
      `gravity_jitter`      — per component, m/s², added to `<option gravity>`.
    """

    mass_scale: Optional[Range] = None
    joint_damping: Optional[Range] = None
    joint_frictionloss: Optional[Range] = None
    kp_scale: Optional[Range] = None
    kd_scale: Optional[Range] = None
    camera_pos_jitter: Optional[Range] = None
    camera_euler_jitter: Optional[Range] = None
    spawn_jitter: Optional[Range] = None
    gravity_jitter: Optional[Range] = None

    def __post_init__(self):
        for f in fields(self):
            r = getattr(self, f.name)
            if r is None:
                continue
            if len(r) != 2 or not all(np.isfinite(v) for v in r):
                raise ValueError(f"{f.name} must be a finite (low, high) pair, got {r!r}")
            lo, hi = float(r[0]), float(r[1])
            if lo > hi:
                raise ValueError(f"{f.name}: low {lo} > high {hi}")
            if f.name in _MULTIPLIERS and lo <= 0.0:
                raise ValueError(f"{f.name} is a multiplier range; low must be > 0, got {lo}")
            if f.name in ("joint_damping", "joint_frictionloss") and lo < 0.0:
                raise ValueError(f"{f.name} must be non-negative, got low {lo}")

    def has_model_params(self) -> bool:
        """True when a draw from this spec needs an MJCF rebuild (see MODEL_KEYS)."""
        return any(
            getattr(self, field) is not None
            for field, key, _ in _FIELDS
            if key in MODEL_KEYS
        )


def sample(spec: RandomizationSpec, rng, ndof: int) -> dict:
    """Draw one randomization from `spec` → a plain, json-serializable dict
    (the `RandomizationDraw`): only the spec'd fields appear as keys, values
    are python floats / lists of floats — diffable and loggable as-is.

    `rng` is a `numpy.random.Generator` or an int seed; an int is expanded via
    `default_rng(seed)`, so `sample(spec, 7, n) == sample(spec, 7, n)` EXACTLY
    (the determinism contract). Draw order is fixed by `_FIELDS` regardless of
    which fields are None, so enabling a field never reshuffles the others'
    values relative to a fresh stream.
    """
    if ndof < 1:
        raise ValueError(f"ndof must be >= 1, got {ndof}")
    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    draw: dict = {}
    for field, key, kind in _FIELDS:
        r = getattr(spec, field)
        if r is None:
            continue
        lo, hi = float(r[0]), float(r[1])
        if kind == "scalar":
            draw[key] = float(rng.uniform(lo, hi))
        else:
            n = ndof if kind == "dof" else 3
            draw[key] = [float(v) for v in rng.uniform(lo, hi, size=n)]
    return draw


def _check_keys(draw: dict) -> None:
    unknown = set(draw) - MODEL_KEYS - ENV_KEYS
    if unknown:
        raise ValueError(f"unknown randomization draw key(s): {sorted(unknown)}")


def _fmt(v: float) -> str:
    return repr(float(v))  # shortest round-trip decimal


def _per_joint(draw: dict, key: str, njoint: int) -> Optional[list[float]]:
    vals = draw.get(key)
    if vals is None:
        return None
    if len(vals) != njoint:
        raise ValueError(f"draw['{key}'] has {len(vals)} entries for {njoint} robot joints")
    return [float(v) for v in vals]


def apply_to_mjcf(draw: dict, mjcf: str) -> str:
    """Apply the MODEL-level fields of `draw` to a `caliper.model_to_mjcf`
    document and return the edited MJCF string. ENV-level keys are ignored
    here (they never touch the compiled model); unknown keys raise.

    The edit is structural (ElementTree, not regex) and touches EXACTLY the
    targeted attributes — robot bodies are the exporter's `b_<joint>` elements
    in document order (= qpos order), so per-joint lists index the same dof
    the rest of caliper does. Mass scaling also scales the body's inertia
    tensor (`fullinertia`/`diaginertia`): with fixed geometry, inertia is
    proportional to mass — scaling one without the other would fabricate a
    physically impossible body.
    """
    _check_keys(draw)
    root = ET.fromstring(mjcf)

    if "gravity" in draw:
        off = draw["gravity"]
        if len(off) != 3:
            raise ValueError(f"draw['gravity'] must have 3 components, got {len(off)}")
        opt = root.find("option")
        if opt is None or opt.get("gravity") is None:
            raise ValueError("MJCF has no <option gravity=...> to jitter")
        base = [float(v) for v in opt.get("gravity").split()]
        opt.set("gravity", " ".join(_fmt(b + float(o)) for b, o in zip(base, off)))

    # Robot bodies in document order (the exporter emits them topologically,
    # so this order IS the caliper q order).
    bodies = [b for b in root.iter("body") if b.get("name", "").startswith("b_")]
    masses = _per_joint(draw, "mass_scale", len(bodies))
    damping = _per_joint(draw, "joint_damping", len(bodies))
    friction = _per_joint(draw, "joint_frictionloss", len(bodies))

    for i, body in enumerate(bodies):
        if masses is not None:
            inertial = body.find("inertial")
            if inertial is None:
                raise ValueError(f"body '{body.get('name')}' has no <inertial> to scale")
            s = masses[i]
            inertial.set("mass", _fmt(float(inertial.get("mass")) * s))
            for attr in ("fullinertia", "diaginertia"):
                raw = inertial.get(attr)
                if raw is not None:
                    inertial.set(attr, " ".join(_fmt(float(v) * s) for v in raw.split()))
        if damping is not None or friction is not None:
            joint = body.find("joint")
            if joint is None:
                raise ValueError(f"body '{body.get('name')}' has no <joint> child")
            if damping is not None:
                joint.set("damping", _fmt(damping[i]))
            if friction is not None:
                joint.set("frictionloss", _fmt(friction[i]))

    return ET.tostring(root, encoding="unicode")


# ----- runtime application ------------------------------------------------------


def _euler_to_quat(e) -> np.ndarray:
    """XYZ (roll-pitch-yaw) euler → wxyz quaternion; composed q = qz * qy * qx
    (rotate about x first) — the small-jitter convention, order-stable."""
    half = 0.5 * np.asarray(e, dtype=np.float64)
    cx, cy, cz = np.cos(half)
    sx, sy, sz = np.sin(half)
    return np.array(
        [
            cz * cy * cx + sz * sy * sx,
            cz * cy * sx - sz * sy * cx,
            cz * sy * cx + sz * cy * sx,
            sz * cy * cx - cz * sy * sx,
        ]
    )


def _quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _apply_camera(scene, pos_off, euler_off) -> None:
    """Jitter the scene camera around its ORIGINAL pose (snapshotted on first
    use, so repeated applications never accumulate)."""
    import mujoco  # lazy: keep this module importable without mujoco

    cid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, scene.camera)
    if cid < 0:
        raise ValueError(f"scene has no camera named {scene.camera!r}")
    base = getattr(scene, "_rand_base_cam", None)
    if base is None:
        base = (scene.model.cam_pos[cid].copy(), scene.model.cam_quat[cid].copy())
        scene._rand_base_cam = base
    if pos_off is not None:
        scene.model.cam_pos[cid] = base[0] + np.asarray(pos_off, dtype=np.float64)
    if euler_off is not None:
        scene.model.cam_quat[cid] = _quat_mul(base[1], _euler_to_quat(euler_off))


def apply_to_env(draw: dict, env, index: int = 0) -> None:
    """Apply the ENV-level fields of `draw` to `env` (a `VecSimEnv`), env
    `index`: PD-gain scales, camera pose jitter, spawn offset. MODEL-level
    keys are ignored here (`apply_to_mjcf` + recompile is their path);
    unknown keys raise.

    Called by `VecSimEnv._reset_env` right before its `mj_forward`; standalone
    callers that pass `spawn_offset` must re-run `mj_forward` themselves. The
    spawn offset ADDS to the current qpos and clips to the env's sampling
    bounds. Camera jitter needs `obs_images=True`; without scenes it is a
    silent no-op (there is no camera to move — documented, not hidden).
    """
    _check_keys(draw)
    if not 0 <= index < env.num_envs:
        raise ValueError(f"env index {index} out of range for num_envs={env.num_envs}")
    if "kp_scale" in draw:
        env._kp[index] = env.kp * float(draw["kp_scale"])
    if "kd_scale" in draw:
        env._kd[index] = env.kd * float(draw["kd_scale"])
    if "spawn_offset" in draw:
        off = np.asarray(draw["spawn_offset"], dtype=np.float64)
        if off.shape != (env.ndof,):
            raise ValueError(f"spawn_offset shape {off.shape} != ({env.ndof},)")
        d = env._data[index]
        d.qpos[:] = np.clip(d.qpos + off, env._bounds[:, 0], env._bounds[:, 1])
    if ("camera_pos" in draw or "camera_euler" in draw) and env._scenes is not None:
        _apply_camera(env._scenes[index], draw.get("camera_pos"), draw.get("camera_euler"))
