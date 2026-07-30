"""`model_to_mjcf(props=...)` — the structured prop face (Wave C).

Free-floating props used to be reachable from Python only as hand-written
`extra_xml`, which meant the caller wrote the `<freejoint>` body, the inertial
and the geom by hand — three chances to get inertia wrong, and props landed
BEFORE the robot bodies (shifting the robot's qpos block). The kwarg routes
the same dicts a `*.caliper-task.json` scene carries through the engine's own
prop rulebook, so there is one spelling and one validator.

What is pinned here: emission ORDER (props after the robot bodies, extra_xml
before them — the qpos-prefix guarantee), the inertia the engine computes,
per-prop materials, and that every malformed spec raises with the engine's own
text rather than producing a plausible-looking document. MuJoCo compiles the
result when it is installed (the qpos layout claim is checked against the real
compiler, not against the XML).
"""

import pathlib
import re

import pytest

caliper = pytest.importorskip("caliper")

FIX = pathlib.Path("oracle/fixtures/robots")

CUBE = {
    "name": "cube",
    "kind": "box",
    "halfExtents": [0.025, 0.025, 0.025],
    "pos": [0.3, 0.0, 0.025],
    "mass": 0.05,
}


def _robot(name="gripper_arm"):
    return caliper.Robot.from_urdf(str(FIX / f"{name}.urdf"))


def _numbers(xml: str, body: str, attr: str) -> list[float]:
    """The floats in `attr` of the first element inside `<body name=body ...>`
    that carries it. Compared as NUMBERS: Rust and Python spell exponents
    differently (`1.6e-5` vs `1.6e-05`), and the values are the claim."""
    tail = xml[xml.index(f'<body name="{body}"') :]
    m = re.search(rf'{attr}="([^"]*)"', tail)
    assert m is not None, f"no {attr} on body {body}"
    return [float(x) for x in m.group(1).split()]


def test_props_emit_freejoint_bodies_after_the_robot():
    r = _robot()
    plain = caliper.model_to_mjcf(r, ground=0.0)
    assert "freejoint" not in plain  # the default is untouched

    xml = caliper.model_to_mjcf(r, ground=0.0, props=[CUBE])
    assert '<body name="prop_cube"' in xml
    assert '<freejoint name="prop_cube_free"/>' in xml
    # AFTER the robot bodies: that is what keeps the robot's qpos the PREFIX.
    assert xml.index('<body name="b_j1"') < xml.index('<body name="prop_cube"')
    # Inertia is computed from mass + shape, never defaulted: a 0.05 kg cube of
    # half-extent a has I = m/3 * (a^2 + a^2) = 2/3 * m * a^2 on every axis.
    i = 2.0 / 3.0 * 0.05 * 0.025**2
    assert _numbers(xml, "prop_cube", "diaginertia") == pytest.approx([i, i, i])
    assert _numbers(xml, "prop_cube", "mass") == pytest.approx([0.05])


def test_props_and_extra_xml_compose_on_opposite_sides_of_the_robot():
    r = _robot()
    xml = caliper.model_to_mjcf(
        r,
        ground=0.0,
        extra_xml='<camera name="ots" pos="1 -1 1"/>',
        props=[CUBE],
    )
    assert xml.index('name="ots"') < xml.index('<body name="b_j1"')
    assert xml.index('<body name="b_j1"') < xml.index('<body name="prop_cube"')


def test_prop_material_rides_through_and_overrides_the_document_material():
    r = _robot()
    # A per-prop preset stamps THAT geom (wood: the documented knobs).
    xml = caliper.model_to_mjcf(r, ground=0.0, props=[{**CUBE, "material": "wood"}])
    prop_geom = xml[xml.index('name="prop_cube_geom"') :]
    assert 'friction="0.45 0.005 0.0001"' in prop_geom
    assert "solref" not in xml[: xml.index('<body name="prop_cube"')]  # robot untouched

    # A custom dict is the same second form the `material=` kwarg takes, and it
    # wins over the document-wide material for that prop.
    xml = caliper.model_to_mjcf(
        r,
        ground=0.0,
        material="steel",
        props=[
            {
                **CUBE,
                "material": {
                    "solref": (0.004, 1.0),
                    "solimp": (0.9, 0.95, 0.001),
                    "friction": (0.7, 0.005, 0.0001),
                },
            }
        ],
    )
    prop_geom = xml[xml.index('name="prop_cube_geom"') :]
    assert 'solref="0.004 1.0"' in prop_geom and 'friction="0.7 0.005 0.0001"' in prop_geom
    assert 'solref="0.004 1.0"' not in xml[: xml.index('<body name="prop_cube"')]


def test_every_prop_kind_and_the_optional_keys():
    r = _robot()
    xml = caliper.model_to_mjcf(
        r,
        props=[
            {"name": "ball", "kind": "sphere", "radius": 0.02, "pos": (0.2, 0.1, 0.02)},
            {
                "name": "can",
                "kind": "cylinder",
                "radius": 0.03,
                "length": 0.1,
                "pos": [0.0, 0.2, 0.05],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "rgba": [0.2, 0.4, 0.9, 1.0],
            },
        ],
    )
    assert 'type="sphere" size="0.02"' in xml
    # MJCF cylinder size is radius + HALF length.
    assert 'type="cylinder" size="0.03 0.05"' in xml
    assert 'rgba="0.2 0.4 0.9 1.0"' in xml
    # An omitted mass takes the documented 0.1 kg default (sphere: 2/5 m r^2).
    assert _numbers(xml, "prop_ball", "mass") == pytest.approx([0.1])
    i = 0.4 * 0.1 * 0.02**2
    assert _numbers(xml, "prop_ball", "diaginertia") == pytest.approx([i, i, i])


@pytest.mark.parametrize(
    "props, message",
    [
        ([{"name": "c", "kind": "cone", "radius": 0.1, "pos": [0, 0, 1]}], "unknown kind"),
        # A typo must not silently produce a default-sized box.
        ([{"name": "c", "kind": "box", "halfExtent": [1, 1, 1], "pos": [0, 0, 1]}], "unknown field"),
        ([{"name": "c", "kind": "box", "pos": [0, 0, 1]}], "box needs halfExtents"),
        ([{"name": "c", "kind": "sphere", "pos": [0, 0, 1]}], "sphere needs radius"),
        ([{"name": "c", "kind": "cylinder", "radius": 0.1, "pos": [0, 0, 1]}], "cylinder needs length"),
        ([{"name": "c", "kind": "sphere", "radius": 0.1}], "missing field `pos`"),
        ([{**CUBE, "mass": 0.0}], "mass must be finite and > 0"),
        ([{**CUBE, "mass": -1.0}], "mass must be finite and > 0"),
        ([{"name": "c", "kind": "sphere", "radius": -0.1, "pos": [0, 0, 1]}], "dimensions must be finite"),
        ([{**CUBE, "pos": [float("nan"), 0, 1]}], "non-finite number"),
        ([{**CUBE, "material": "granite"}], "unknown material preset"),
        ([{**CUBE, "material": {"solref": [0.01, 1.0]}}], "missing `solimp`"),
        ([CUBE, {"name": "cube", "kind": "sphere", "radius": 0.01, "pos": [0, 0, 1]}], "duplicate prop name"),
        (CUBE, "must be a list of prop dicts"),  # one dict, not a list of them
    ],
)
def test_malformed_props_raise_with_the_engine_text(props, message):
    with pytest.raises(ValueError, match=message):
        caliper.model_to_mjcf(_robot(), props=props)


def test_mujoco_compiles_props_behind_the_robot_dofs():
    mujoco = pytest.importorskip("mujoco")
    r = _robot()
    xml = caliper.model_to_mjcf(
        r,
        ground=0.0,
        props=[CUBE, {"name": "ball", "kind": "sphere", "radius": 0.02, "pos": [0.2, 0.1, 0.02]}],
    )
    m = mujoco.MjModel.from_xml_string(xml)
    # Two free joints behind the arm: 7 qpos / 6 qvel each, robot first.
    assert m.nq == r.ndof + 2 * 7 and m.nv == r.ndof + 2 * 6
    for i, name in enumerate(r.joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert m.jnt_qposadr[jid] == i, "the robot's qpos block is not the prefix"
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "prop_cube")
    assert list(d.xpos[bid]) == pytest.approx(CUBE["pos"])
