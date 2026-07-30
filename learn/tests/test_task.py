"""`*.caliper-task.json` in python — the loader, and the RUST↔PYTHON PARITY TABLE.

The task artifact has two readers: `caliper_sim_mujoco::task` (Studio, the CLI,
anything driving a live sim) and `caliper_learn.task` (rollouts, eval, the
autopsy). A file scored differently by the two would mean a Studio teleop take
and an eval report can disagree about whether the same episode succeeded, which
is the exact failure the shared schema exists to prevent.

So the same three predicates, the same six hand-computed states and the same
expected verdicts are asserted here and in `task.rs`'s
`parity_table_with_the_python_evaluator` — the table below is a transcription of
that test, cell for cell. Two of its rows sit deliberately ON a boundary,
because that is where two implementations drift: `nudged` clears 5 mm of a 50 mm
bar, and `on face` sits exactly on the bin's +x face at exactly the settle speed
(both INSIDE — faces and the speed limit are inclusive, and
|0.45 - 0.4| is 0.04999999999999999 in binary, which is BELOW 0.05, not on it).
Both faces do the same binary arithmetic with no epsilon anywhere, so both must
agree.

The rest pins the loader: both shipped fixtures load, the strict rules bite
(unknown keys, bad version, missing prop, undeclared zone, duplicate names,
negative extents, a robot path that does not resolve), zone-by-NAME resolves
against the scene exactly as Rust's does, and `to_dict` round-trips.

Pure python: no mujoco, no caliper (the loader touches neither) — `VecSimEnv`
integration lives in test_vec_env.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from caliper_learn.success import SuccessState
from caliper_learn.success import from_dict as success_from_dict
from caliper_learn.task import TASK_VERSION, load_task, task_from_dict

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "oracle" / "fixtures" / "tasks"
PICK = TASKS / "pick_cube.caliper-task.json"
LIFT = TASKS / "lift_cube.caliper-task.json"


def _raw(path: Path) -> dict:
    return json.loads(path.read_text())


# A minimal valid task — the base every error case mutates (the same one
# task.rs's `base_task` uses, so the two error tables line up too).
def _base() -> dict:
    return {
        "version": 1,
        "name": "t",
        "robot": "../robots/gripper_arm.urdf",
        "scene": {
            "props": [
                {
                    "name": "cube",
                    "kind": "box",
                    "halfExtents": [0.05, 0.05, 0.05],
                    "pos": [0.0, 0.0, 0.05],
                    "mass": 0.05,
                }
            ],
            "zones": [
                {"name": "bin", "center": [0.4, 0.2, 0.02], "half": [0.05, 0.05, 0.02]}
            ],
        },
    }


def _load(raw: dict):
    return task_from_dict(raw, base_dir=TASKS)


# ----- the fixtures ------------------------------------------------------------


def test_pick_cube_fixture_loads_and_round_trips():
    task = load_task(PICK)
    assert task.name == "pick-cube"
    assert task.robot == "../robots/gripper_arm.urdf"
    assert task.robot_path.is_file()
    assert task.q0 == [0.0, 0.0, 0.02]
    assert task.ground == 0.0 and task.ground_height() == 0.0
    assert task.prop_names() == ["cube"]
    assert task.props[0]["material"] == "wood"
    assert task.zones[0]["name"] == "bin"
    assert task.gripper == {"joint": "gripper", "closed": "lo"}
    assert task.fps == 50 and task.horizon_s == 20.0
    # 20 s at 50 Hz is 1000 control steps — the conversion an eval budget uses.
    assert task.steps(50) == 1000 and task.steps(30) == 600

    # The fixture's success zone is spelled out, so nothing is resolved away and
    # the round-trip is exact (the Rust test asserts the same equality).
    assert task.to_dict() == _raw(PICK)
    assert _load(task.to_dict()).to_dict() == task.to_dict()


def test_pick_cube_success_matches_the_rust_wording_and_verdicts():
    task = load_task(PICK)
    assert task.success.describe() == (
        "cube's center is inside the 0.100 x 0.100 x 0.040 m box centered at "
        "(0.400, 0.200, 0.020) and has come to rest (speed <= 0.010 m/s)"
    )
    at_rest = SuccessState(
        prop_pos={"cube": (0.4, 0.2, 0.02)}, prop_vel={"cube": (0.0, 0.0, 0.0)}
    )
    flying = SuccessState(
        prop_pos={"cube": (0.4, 0.2, 0.02)}, prop_vel={"cube": (0.0, 0.0, -1.5)}
    )
    assert task.success(at_rest) is True
    assert task.success(flying) is False


def test_lift_cube_fixture_resolves_its_zone_by_name():
    task = load_task(LIFT)
    spec = task.success.to_dict()
    assert spec["kind"] == "all_of"
    assert spec["terms"][0] == {
        "kind": "lifted",
        "prop": "cube",
        "height": 0.05,
        "ref": "initial",
    }
    # The file writes `"zone": "bin"`; what we hold is the resolved box — and
    # `to_dict` carries the RESOLVED form (the documented round-trip exception).
    bin_zone = task.zone("bin")
    assert spec["terms"][1]["zone"] == {
        "center": bin_zone["center"],
        "half": bin_zone["half"],
    }
    assert _raw(LIFT)["success"]["terms"][1]["zone"] == "bin"
    assert task.to_dict()["success"] == spec
    # Re-loading the serialized form is a no-op.
    assert _load(task.to_dict()).to_dict() == task.to_dict()
    # No horizon in this file: the reader says so instead of inventing one.
    assert task.horizon_s is None and task.steps(50) is None


def test_a_scene_only_task_needs_no_success():
    task = _load({"version": 1, "name": "free", "robot": "../robots/gripper_arm.urdf", "scene": {}})
    assert task.success is None
    assert task.props == [] and task.zones == []
    assert task.ground is None and task.ground_height() == 0.0
    assert task.to_dict()["scene"] == {}


# ----- the parity table --------------------------------------------------------

# The three predicates, byte-for-byte the JSON task.rs uses.
A = {"kind": "lifted", "prop": "cube", "height": 0.05, "ref": "initial"}
B = {
    "kind": "placed_in_zone",
    "prop": "cube",
    "zone": {"center": [0.4, 0.2, 0.02], "half": [0.05, 0.05, 0.02]},
    "settled_speed": 0.01,
}
C = {
    "kind": "any_of",
    "terms": [
        {"kind": "lifted", "prop": "cube", "height": 0.05, "ref": "absolute"},
        {
            "kind": "placed_in_zone",
            "prop": "cube",
            "zone": {"center": [0.4, 0.2, 0.02], "half": [0.05, 0.05, 0.02]},
            "settled_speed": None,
        },
    ],
}

SPAWN = (0.3, 0.0, 0.025)  # on the table, where S0 is
NUDGE = (0.3, 0.0, 0.03)  # 5 mm above SPAWN
HIGH = (0.3, 0.0, 0.2)  # carried well clear
IN_BIN = (0.4, 0.2, 0.02)  # the bin's center
ON_FACE = (0.45, 0.2, 0.02)  # exactly the bin's +x face
REST = (0.0, 0.0, 0.0)
RISING = (0.0, 0.0, 0.1)
FAST_UP = (0.0, 0.0, 0.5)
FALLING = (0.0, 0.0, -1.5)
AT_LIMIT = (0.01, 0.0, 0.0)  # exactly settled_speed

# (label, pos, vel, [A, B, C]) — transcribed from task.rs.
PARITY = [
    ("table", SPAWN, REST, [False, False, False]),
    ("nudged", NUDGE, RISING, [False, False, False]),
    ("carried", HIGH, FAST_UP, [True, False, True]),
    ("placed", IN_BIN, REST, [False, True, True]),
    ("flying", IN_BIN, FALLING, [False, False, True]),
    ("on face", ON_FACE, AT_LIMIT, [False, True, True]),
]


def _state(pos, vel) -> SuccessState:
    return SuccessState(prop_pos={"cube": pos}, prop_vel={"cube": vel})


@pytest.mark.parametrize("label, pos, vel, want", PARITY, ids=[r[0] for r in PARITY])
def test_parity_table_with_the_rust_evaluator(label, pos, vel, want):
    # A needs an episode baseline (the reset state IS the spawn pose); B and C
    # are pure functions of the state.
    s0 = _state(SPAWN, REST)
    state = _state(pos, vel)
    a = success_from_dict(A)
    a.reset(s0)
    assert bool(a(state)) is want[0], f"A on {label!r}"
    assert bool(success_from_dict(B)(state)) is want[1], f"B on {label!r}"
    assert bool(success_from_dict(C)(state)) is want[2], f"C on {label!r}"


def test_the_two_parity_error_rows():
    b = success_from_dict(B)
    # A settled check with no velocities must ERROR, not pass.
    with pytest.raises(ValueError, match="no velocities"):
        b(SuccessState(prop_pos={"cube": IN_BIN}))
    # A missing prop must ERROR, not pass.
    with pytest.raises(ValueError, match="no prop named 'cube'"):
        b(SuccessState(prop_pos={"ball": IN_BIN}, prop_vel={"ball": REST}))


def test_the_knife_edges_the_table_rides_on():
    """Why the boundary rows read the way they do — stated as arithmetic so a
    future epsilon added to either face fails HERE, not mysteriously above."""
    assert abs(0.45 - 0.4) == 0.04999999999999999  # INSIDE a 0.05 half-extent
    assert abs(0.45 - 0.4) <= 0.05
    assert math.sqrt(0.01**2) == 0.01  # exactly AT the settle limit, inclusive
    assert (0.15 - 0.10) < 0.05  # the documented non-lift: 0.049999999999999996


# ----- strictness --------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: t.update(hoirzon=3.0), "hoirzon"),
        (lambda t: t.update(Name="x"), "Name"),
        (lambda t: t["scene"].update(gravity=9.8), "gravity"),
        (lambda t: t["scene"]["props"][0].update(halfExtent=[1, 1, 1]), "halfExtent"),
        (lambda t: t["scene"]["zones"][0].update(colour=[1, 1, 1, 1]), "colour"),
        (lambda t: t.update(gripper={"joint": "g", "close": "lo"}), "close"),
        (
            lambda t: t["scene"]["props"][0].update(
                material={"solref": [0.01, 1.0], "solimp": [0.9, 0.95, 0.001], "frictoin": [1, 0, 0]}
            ),
            "frictoin",
        ),
        (
            lambda t: t.update(
                success={"kind": "lifted", "prop": "cube", "height": 0.05, "reff": "initial"}
            ),
            "reff",
        ),
    ],
    ids=[
        "top-level",
        "top-level-case",
        "scene",
        "prop",
        "zone",
        "gripper",
        "material",
        "success",
    ],
)
def test_unknown_keys_raise_at_every_level(mutate, message):
    raw = _base()
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        _load(raw)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: t.update(version=2), "version 2 is not supported"),
        (lambda t: t.update(version="1"), "not supported"),
        (lambda t: t.update(version=1.0), "not supported"),  # an int, like Rust's u32
        (lambda t: t.update(version=True), "not supported"),  # True == 1 in python
        (lambda t: t.update(name="  "), "`name` must be a non-empty"),
        (lambda t: t.update(robot="../robots/nope.urdf"), "does not resolve to a file"),
        (lambda t: t.update(robot=""), "must be a non-empty path"),
        (lambda t: t.update(q0=[]), "`q0` must be a non-empty"),
        (lambda t: t.update(q0=[0.0, None]), "`q0` must be"),
        (lambda t: t.update(horizonS=0.0), "must be finite and > 0"),
        (lambda t: t.update(horizonS=float("inf")), "must be finite"),
        (lambda t: t.update(fps=0), "`fps` must be an int > 0"),
        (lambda t: t.update(gripper={"joint": "gripper", "closed": "shut"}), 'must be "lo" or "hi"'),
        (lambda t: t.update(gripper={"joint": "  "}), "non-empty joint name"),
        (lambda t: t["scene"]["props"][0].update(mass=0.0), "mass must be finite and > 0"),
        (lambda t: t["scene"]["props"][0].update(mass=-1.0), "mass must be finite and > 0"),
        (lambda t: t["scene"]["props"][0].update(kind="cone"), "unknown kind"),
        (
            lambda t: t["scene"]["props"][0].pop("halfExtents") and None,
            "box needs halfExtents",
        ),
        (lambda t: t["scene"]["props"][0].update(halfExtents=[0.05, 0.0, 0.05]), "must be > 0"),
        (lambda t: t["scene"]["props"][0].update(pos=[0.0, 0.0]), "pos must be 3 numbers"),
        (
            lambda t: t["scene"]["props"][0].update(quat=[0.0, 0.0, 0.0, 0.0]),
            "non-zero norm",
        ),
        (lambda t: t["scene"]["props"][0].update(material="granite"), "unknown material preset"),
        (
            lambda t: t["scene"]["props"][0].update(material={"solref": [0.01, 1.0]}),
            "missing `solimp`",
        ),
        (lambda t: t["scene"]["zones"][0].update(half=[0.05, -0.05, 0.02]), "non-negative"),
        (
            lambda t: t["scene"]["zones"].append(
                {"name": "bin", "center": [0, 0, 0], "half": [1, 1, 1]}
            ),
            "duplicate zone name",
        ),
        (
            lambda t: t["scene"]["props"].append(
                {"name": "cube", "kind": "sphere", "radius": 0.01, "pos": [1, 1, 1]}
            ),
            "duplicate prop name",
        ),
        (
            # The whitespace rule: `cube` and `cu be` are ONE MuJoCo body name.
            lambda t: t["scene"]["props"].append(
                {"name": "cu be", "kind": "sphere", "radius": 0.01, "pos": [1, 1, 1]}
            ),
            None,  # legal: 'cu_be' != 'cube'
        ),
        (
            lambda t: t.update(success={"kind": "lifted", "prop": "ball", "height": 0.05}),
            "which the scene does not contain",
        ),
        (
            lambda t: t.update(
                success={"kind": "placed_in_zone", "prop": "cube", "zone": "crate"}
            ),
            "does not declare",
        ),
        (lambda t: t.pop("scene"), "needs a `scene`"),
        (lambda t: t.update(scene={"props": {}}), "must be a list"),
    ],
)
def test_validation_refuses_impossible_tasks(mutate, message):
    raw = _base()
    mutate(raw)
    if message is None:
        _load(raw)  # the negative control: this one IS legal
        return
    with pytest.raises(ValueError, match=message):
        _load(raw)


def test_a_named_zone_deeper_in_the_tree_resolves():
    raw = _base()
    raw["success"] = {
        "kind": "any_of",
        "terms": [
            {"kind": "lifted", "prop": "cube", "height": 0.05},
            {
                "kind": "all_of",
                "terms": [
                    {"kind": "placed_in_zone", "prop": "cube", "zone": "bin", "settled_speed": 0.01}
                ],
            },
        ],
    }
    task = _load(raw)
    inner = task.success.to_dict()["terms"][1]["terms"][0]
    assert inner["zone"] == {"center": [0.4, 0.2, 0.02], "half": [0.05, 0.05, 0.02]}
    # ... and the resolved predicate is answerable (a named zone never is).
    state = _state(IN_BIN, REST)
    assert bool(task.success(state)) is True


def test_load_task_reports_the_file_it_could_not_read(tmp_path):
    with pytest.raises(ValueError, match="reading task file"):
        load_task(tmp_path / "nope.caliper-task.json")
    bad = tmp_path / "bad.caliper-task.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not a valid caliper task file"):
        load_task(bad)


def test_an_absolute_robot_path_is_taken_as_is(tmp_path):
    raw = _base()
    raw["robot"] = str((TASKS / "../robots/gripper_arm.urdf").resolve())
    task = task_from_dict(raw, base_dir=tmp_path)  # base_dir must be ignored
    assert task.robot_path.is_file()
    assert task.to_dict()["robot"] == raw["robot"]


def test_the_version_constant_is_the_one_the_files_carry():
    assert TASK_VERSION == 1
    assert _raw(PICK)["version"] == TASK_VERSION
    assert _raw(LIFT)["version"] == TASK_VERSION
