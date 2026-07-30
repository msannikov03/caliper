"""`*.caliper-task.json` — a manipulation TASK as a file, read into python.

A task artifact is the small, diffable thing that turns "drive the arm around"
into "do THIS, and here is what counts as done": which robot, the scene it acts
on (free props with contact materials, plus target ZONES that exist only for
the evaluator and the renderer), which joint is the gripper, and the success
criterion — the very same predicate schema `success.py` reads. One file scores
a Studio teleop take, a sim rollout and an eval report identically.

This module is the PYTHON reader of the schema `caliper_sim_mujoco::task`
defines (`crates/caliper-sim-mujoco/src/task.rs`); the two are a port, not
parallel inventions, and `learn/tests/test_task.py` runs the same parity table
as the Rust test so the faces cannot drift.

    task = load_task("pick_cube.caliper-task.json")
    env = VecSimEnv.from_task(task, num_envs=4)     # scene + success + q0
    result = evaluate(policy, eval_task_from_file(path))

# The rules, and why they are strict

- `version` must be `TASK_VERSION`. A future file is refused, never guessed at.
- UNKNOWN KEYS RAISE — at the top level and inside every nested object. A
  `settle_speed` typo that silently dropped the settle requirement, or a
  `halfExtent` that silently produced a default-sized box, is exactly the class
  of bug this file format exists to make impossible. Same rule, same reason, as
  `success.from_dict`.
- `robot` resolves against the TASK FILE's own directory (or is absolute) and
  must exist: a task naming a robot that is not there has no honest reading.
- Inside `success`, a zone written as a STRING is resolved by name against
  `scene.zones` when the file loads, and the RESOLVED form is what `to_dict`
  writes back — so a task file may name a zone once while every predicate
  handed to another layer carries real numbers. `success.from_dict` itself
  keeps accepting only the resolved form, which is why resolution happens here.
- Props are kept as the DICTS they were written as, in the spelling
  `caliper.model_to_mjcf(props=...)` takes, so the scene reaches MuJoCo through
  the engine's own prop rulebook (one implementation of prop inertia and
  material handling, not a second copy in python).

# Where validation lives (the one deliberate seam)

Everything structural is checked here: keys, version, required-by-kind
dimensions, duplicate names, zone geometry, `q0`/`horizonS`/`fps`/`gripper`,
the success cross-check that every scored prop is actually in the scene, and
the definitional prop values (a mass must be positive and finite, a dimension
likewise — that is what the words mean). What is NOT re-implemented here is the
engine's contact-material rulebook (the admissible `solref`/`solimp`/`friction`
RANGES) and the inertia it computes: material SHAPE is checked here, its values
are rejected by `caliper.model_to_mjcf(props=...)` with the engine's own text
when the scene is built. Rust's `load_task` applies the material ranges at load
time, so a task with an out-of-range solver knob is refused one step later in
python than in Rust — the only asymmetry between the two readers.

Pure python: no mujoco, no torch, no caliper (`success` is the only import).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .success import SuccessPredicate, Zone
from .success import from_dict as success_from_dict

# The only schema version this module reads or writes.
TASK_VERSION = 1

# Prop mass used when the file omits one (kg) — the value the engine's
# `TaskProp` applies, restated here only so the docs can name it.
DEFAULT_PROP_MASS = 0.1

_TOP_KEYS = {
    "version",
    "name",
    "robot",
    "q0",
    "scene",
    "gripper",
    "success",
    "horizonS",
    "fps",
}
_SCENE_KEYS = {"ground", "props", "zones"}
_PROP_KEYS = {
    "name",
    "kind",
    "halfExtents",
    "radius",
    "length",
    "pos",
    "quat",
    "mass",
    "rgba",
    "material",
}
_ZONE_KEYS = {"name", "center", "half", "rgba"}
_GRIPPER_KEYS = {"joint", "closed"}
_MATERIAL_KEYS = {"solref", "solimp", "friction"}
_MATERIAL_PRESETS = ("rigid", "rubber", "foam", "steel", "wood")

# Which dimension key(s) each prop kind requires (the engine's own dispatch).
_KIND_DIMS = {
    "box": ("halfExtents",),
    "sphere": ("radius",),
    "cylinder": ("radius", "length"),
}


def _mujoco_name(name: str) -> str:
    """The identifier MuJoCo actually REGISTERS for a caliper name: whitespace
    becomes `_`. Mirrors `caliper_sim_mujoco::mjcf::mujoco_name`, which is the
    spelling prop bodies collide under — so `"a b"` and `"a_b"` are ONE name."""
    return "".join("_" if ch.isspace() else ch for ch in name)


def _check_keys(d: Mapping, allowed: set, what: str) -> None:
    if not isinstance(d, Mapping):
        raise ValueError(f"{what} must be an object, got {type(d).__name__}")
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(
            f"unknown key(s) in {what}: {sorted(unknown)} (allowed: {sorted(allowed)})"
        )


def _floats(v, n: int, what: str) -> list[float]:
    """`n` finite numbers, or a `ValueError` saying what was there instead."""
    if isinstance(v, (str, bytes, Mapping)) or not isinstance(v, Sequence):
        raise ValueError(f"{what} must be {n} numbers, got {type(v).__name__}")
    if len(v) != n:
        raise ValueError(f"{what} must be {n} numbers, got {len(v)}")
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"{what} must be {n} numbers, got {x!r} in it")
        if not math.isfinite(float(x)):
            raise ValueError(f"{what} must be finite, got {x!r}")
        out.append(float(x))
    return out


def _number(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{what} must be a number, got {type(v).__name__}")
    if not math.isfinite(float(v)):
        raise ValueError(f"{what} must be finite, got {v!r}")
    return float(v)


# ----- the task ----------------------------------------------------------------


@dataclass(frozen=True)
class LearnTask:
    """One manipulation task, as loaded from a `*.caliper-task.json`.

    `robot` is the file's own path string; `robot_path` is that resolved
    against the task file's directory (the one to hand
    `caliper.Robot.from_urdf`). `props` and `zones` are the file's dicts,
    validated — `props` goes straight into
    `caliper.model_to_mjcf(props=...)` / `VecSimEnv(props=...)`. `success` is a
    live `SuccessPredicate` with every zone resolved, or None for a task that
    defines no verdict (a free teleop scene). `ground` is the file's own value,
    None when absent — `ground_height()` is the resolved one.
    """

    name: str
    robot: str
    robot_path: Path
    q0: Optional[list[float]]
    ground: Optional[float]
    props: list[dict]
    zones: list[dict]
    gripper: Optional[dict]
    success: Optional[SuccessPredicate]
    horizon_s: Optional[float]
    fps: Optional[int]

    def ground_height(self) -> float:
        """The ground-plane height, 0.0 when the file omits it (the engine's
        `SceneSpec::ground_height`)."""
        return 0.0 if self.ground is None else self.ground

    def zone(self, name: str) -> Optional[dict]:
        """The named zone dict, if this scene declares it."""
        return next((z for z in self.zones if z["name"] == name), None)

    def prop_names(self) -> list[str]:
        return [p["name"] for p in self.props]

    def steps(self, fps: int) -> Optional[int]:
        """`horizonS` as a number of control steps at `fps` — the honest
        conversion for a step-budgeted runner (`EvalTask.max_steps`).

        Rounded UP so the full time budget is covered, and never below 1: a
        20 s horizon at 50 Hz is 1000 steps. None when the task sets no
        horizon (the caller's own default then applies)."""
        if self.horizon_s is None:
            return None
        if not (isinstance(fps, int) and fps > 0):
            raise ValueError(f"fps must be a positive int, got {fps!r}")
        return max(1, math.ceil(self.horizon_s * fps))

    def to_dict(self) -> dict:
        """The file form. Round-trips through `load_task` with ONE documented
        difference, the same one Rust's `to_json` has: a `success` zone written
        as a NAME comes back spelled out as `{center, half}`."""
        scene: dict[str, Any] = {}
        if self.ground is not None:
            scene["ground"] = self.ground
        if self.props:
            scene["props"] = [dict(p) for p in self.props]
        if self.zones:
            scene["zones"] = [dict(z) for z in self.zones]
        out: dict[str, Any] = {
            "version": TASK_VERSION,
            "name": self.name,
            "robot": self.robot,
        }
        if self.q0 is not None:
            out["q0"] = list(self.q0)
        out["scene"] = scene
        if self.gripper is not None:
            out["gripper"] = dict(self.gripper)
        if self.success is not None:
            out["success"] = self.success.to_dict()
        if self.horizon_s is not None:
            out["horizonS"] = self.horizon_s
        if self.fps is not None:
            out["fps"] = self.fps
        return out

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ----- loading -----------------------------------------------------------------


def load_task(path: str | os.PathLike) -> LearnTask:
    """Read and fully resolve a task file: the robot path relative to the FILE,
    zone names against the file's own scene, every rule in the module doc
    checked. `ValueError` on anything the schema does not allow."""
    p = Path(path)
    try:
        text = p.read_text()
    except OSError as e:
        raise ValueError(f"reading task file {str(p)!r}: {e}") from None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"not a valid caliper task file — {p}: {e}") from None
    return task_from_dict(raw, base_dir=p.parent)


def task_from_dict(raw: Mapping, *, base_dir: str | os.PathLike = ".") -> LearnTask:
    """`load_task` without the file read: resolve relative paths and zone names
    against `base_dir` and the spec's own scene, then validate."""
    _check_keys(raw, _TOP_KEYS, "a task file")

    version = raw.get("version")
    # An INT, exactly like the Rust reader's `u32`: `1.0` and `True` (which
    # equals 1 in python) are not this schema's version, they are typos.
    if not isinstance(version, int) or isinstance(version, bool) or version != TASK_VERSION:
        raise ValueError(
            f"task file version {version!r} is not supported — this build reads "
            f"version {TASK_VERSION}"
        )
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("task `name` must be a non-empty string")
    robot = raw.get("robot")
    if not isinstance(robot, str) or not robot.strip():
        raise ValueError(f"task {name!r}: `robot` must be a non-empty path")
    robot_path = Path(robot)
    if not robot_path.is_absolute():
        robot_path = Path(base_dir) / robot_path
    if not robot_path.is_file():
        raise ValueError(
            f"task {name!r}: robot {robot!r} does not resolve to a file (looked at "
            f"{robot_path})"
        )

    q0 = raw.get("q0")
    if q0 is not None:
        if not isinstance(q0, Sequence) or isinstance(q0, (str, bytes)) or not q0:
            raise ValueError(
                f"task {name!r}: `q0` must be a non-empty list of finite numbers"
            )
        q0 = _floats(q0, len(q0), f"task {name!r}: `q0`")

    horizon_s = raw.get("horizonS")
    if horizon_s is not None:
        horizon_s = _number(horizon_s, f"task {name!r}: `horizonS`")
        if horizon_s <= 0.0:
            raise ValueError(
                f"task {name!r}: `horizonS` must be finite and > 0, got {horizon_s}"
            )
    fps = raw.get("fps")
    if fps is not None:
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError(f"task {name!r}: `fps` must be an int > 0, got {fps!r}")

    gripper = _gripper(raw.get("gripper"), name)
    scene = raw.get("scene")
    if scene is None:
        raise ValueError(f"task {name!r}: a task needs a `scene` (`{{}}` for an empty one)")
    _check_keys(scene, _SCENE_KEYS, f"task {name!r}: `scene`")
    ground = (
        None
        if scene.get("ground") is None
        else _number(scene["ground"], f"task {name!r}: scene `ground`")
    )
    # `is None`, not `or []`: a falsy-but-wrong `"props": {}` must be REFUSED,
    # not read as an empty scene.
    props = _props([] if scene.get("props") is None else scene["props"], name)
    zones = _zones([] if scene.get("zones") is None else scene["zones"], name)
    success = _success(raw.get("success"), name, props, zones)

    return LearnTask(
        name=name,
        robot=robot,
        robot_path=robot_path,
        q0=q0,
        ground=ground,
        props=props,
        zones=zones,
        gripper=gripper,
        success=success,
        horizon_s=horizon_s,
        fps=fps,
    )


def _gripper(spec, task: str) -> Optional[dict]:
    """The gripper-channel override: which joint, and which limit CLOSES it."""
    if spec is None:
        return None
    _check_keys(spec, _GRIPPER_KEYS, f"task {task!r}: `gripper`")
    joint = spec.get("joint")
    if joint is not None and (not isinstance(joint, str) or not joint.strip()):
        raise ValueError(
            f"task {task!r}: gripper `joint` must be a non-empty joint name"
        )
    closed = spec.get("closed")
    if closed is not None and closed not in ("lo", "hi"):
        raise ValueError(
            f"task {task!r}: gripper `closed` must be \"lo\" or \"hi\", got "
            f"{closed!r} — which limit CLOSES the gripper"
        )
    return dict(spec)


def _props(specs, task: str) -> list[dict]:
    """Every prop, validated structurally and definitionally (see the module
    doc's seam note: the engine owns inertia and the material RANGES)."""
    if isinstance(specs, Mapping) or not isinstance(specs, Sequence):
        raise ValueError(
            f"task {task!r}: scene `props` must be a list of prop objects, got "
            f"{type(specs).__name__}"
        )
    out: list[dict] = []
    seen: set[str] = set()
    for i, spec in enumerate(specs):
        _check_keys(spec, _PROP_KEYS, f"task {task!r}: prop[{i}]")
        pname = spec.get("name")
        if not isinstance(pname, str) or not pname.strip():
            raise ValueError(f"task {task!r}: prop[{i}] `name` must be a non-empty string")
        where = f"task {task!r}: prop {pname!r}"
        body = _mujoco_name(pname)
        if body in seen:
            raise ValueError(
                f"task {task!r}: duplicate prop name {pname!r} (after sanitizing)"
            )
        seen.add(body)

        kind = spec.get("kind")
        if kind not in _KIND_DIMS:
            raise ValueError(
                f"{where}: unknown kind {kind!r} ({'|'.join(_KIND_DIMS)})"
            )
        for key in _KIND_DIMS[kind]:
            if spec.get(key) is None:
                raise ValueError(f"{where}: {kind} needs {key}")
        # Positive, finite dimensions: definitional, not an engine convention.
        if kind == "box":
            for h in _floats(spec["halfExtents"], 3, f"{where}: halfExtents"):
                if h <= 0.0:
                    raise ValueError(f"{where}: halfExtents must be > 0, got {h}")
        else:
            for key in _KIND_DIMS[kind]:
                if _number(spec[key], f"{where}: {key}") <= 0.0:
                    raise ValueError(f"{where}: {key} must be > 0, got {spec[key]}")
        if spec.get("pos") is None:
            raise ValueError(f"{where}: missing `pos` (the initial world center)")
        _floats(spec["pos"], 3, f"{where}: pos")
        if spec.get("quat") is not None:
            q = _floats(spec["quat"], 4, f"{where}: quat (w-first)")
            if math.sqrt(sum(x * x for x in q)) < 1e-9:
                raise ValueError(f"{where}: quat must have non-zero norm")
        if spec.get("mass") is not None:
            mass = _number(spec["mass"], f"{where}: mass")
            if mass <= 0.0:
                raise ValueError(f"{where}: mass must be finite and > 0, got {mass}")
        if spec.get("rgba") is not None:
            _floats(spec["rgba"], 4, f"{where}: rgba")
        if spec.get("material") is not None:
            _material(spec["material"], where)
        out.append(dict(spec))
    return out


def _material(spec, where: str) -> None:
    """A contact material's SHAPE: a preset NAME or exactly the three raw
    knobs. The admissible VALUE ranges are the engine's rulebook and are
    enforced when the scene is built (module doc)."""
    if isinstance(spec, str):
        if spec.lower() not in _MATERIAL_PRESETS:
            raise ValueError(
                f"{where}: unknown material preset {spec!r} — expected one of "
                f"{', '.join(_MATERIAL_PRESETS)}, or a custom object "
                "{solref, solimp, friction}"
            )
        return
    _check_keys(spec, _MATERIAL_KEYS, f"{where}: material")
    for key, n in (("solref", 2), ("solimp", 3), ("friction", 3)):
        if spec.get(key) is None:
            raise ValueError(f"{where}: custom material is missing `{key}`")
        _floats(spec[key], n, f"{where}: material {key}")


def _zones(specs, task: str) -> list[dict]:
    """Evaluator-side target regions. Nothing is emitted into MJCF for them."""
    if isinstance(specs, Mapping) or not isinstance(specs, Sequence):
        raise ValueError(
            f"task {task!r}: scene `zones` must be a list of zone objects, got "
            f"{type(specs).__name__}"
        )
    out: list[dict] = []
    names: set[str] = set()
    for i, spec in enumerate(specs):
        _check_keys(spec, _ZONE_KEYS, f"task {task!r}: zone[{i}]")
        zname = spec.get("name")
        if not isinstance(zname, str) or not zname.strip():
            raise ValueError(f"task {task!r}: every zone needs a non-empty name")
        if zname in names:
            raise ValueError(f"task {task!r}: duplicate zone name {zname!r}")
        names.add(zname)
        # Zone's own constructor is the geometry rulebook (finite center/half,
        # non-negative extents) — the same one the predicates use.
        _zone_of(spec, task)
        if spec.get("rgba") is not None:
            _floats(spec["rgba"], 4, f"task {task!r}: zone {zname!r} rgba")
        out.append(dict(spec))
    return out


def _zone_of(spec: Mapping, task: str) -> Zone:
    for key in ("center", "half"):
        if spec.get(key) is None:
            raise ValueError(
                f"task {task!r}: zone {spec.get('name')!r} is missing required key "
                f"{key!r}"
            )
    try:
        return Zone(center=tuple(spec["center"]), half=tuple(spec["half"]))
    except (TypeError, ValueError) as e:
        raise ValueError(f"task {task!r}: zone {spec.get('name')!r}: {e}") from None


def _success(spec, task: str, props: list[dict], zones: list[dict]):
    """Resolve zone NAMES against `zones`, build the predicate, then apply the
    cross-check a task file makes possible: every prop it scores must be in the
    scene."""
    if spec is None:
        return None
    resolved = _resolve_zones(spec, task, zones)
    try:
        predicate = success_from_dict(resolved)
    except (TypeError, ValueError) as e:
        raise ValueError(f"task {task!r}: `success` {e}") from None
    known = [p["name"] for p in props]
    for prop in sorted(_scored_props(resolved)):
        if prop not in known:
            raise ValueError(
                f"task {task!r}: `success` scores prop {prop!r}, which the scene does "
                f"not contain (props: {known})"
            )
    return predicate


def _resolve_zones(spec, task: str, zones: list[dict]):
    """A deep copy of `spec` with every `"zone": "<name>"` replaced by that
    zone's `{center, half}` — the form `success.from_dict` accepts."""
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"task {task!r}: `success` must be a predicate object, got "
            f"{type(spec).__name__}"
        )
    out = {}
    for key, value in spec.items():
        if key == "terms" and isinstance(value, Sequence) and not isinstance(value, str):
            out[key] = [_resolve_zones(t, task, zones) for t in value]
        elif key == "zone" and isinstance(value, str):
            named = next((z for z in zones if z["name"] == value), None)
            if named is None:
                raise ValueError(
                    f"task {task!r}: `success` refers to zone {value!r}, which "
                    f"`scene.zones` does not declare (zones: "
                    f"{[z['name'] for z in zones]})"
                )
            out[key] = _zone_of(named, task).to_dict()
        else:
            out[key] = value
    return out


def _scored_props(spec: Mapping) -> set[str]:
    """Every prop name a predicate spec reads, at any depth."""
    terms = spec.get("terms")
    if isinstance(terms, Sequence) and not isinstance(terms, str):
        out: set[str] = set()
        for t in terms:
            if isinstance(t, Mapping):
                out |= _scored_props(t)
        return out
    prop = spec.get("prop")
    return {prop} if isinstance(prop, str) else set()
