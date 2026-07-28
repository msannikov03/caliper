"""Oracle smokes for the 2026-07-28 surface cleanups — the previously
faceless engine capabilities get their faces, and the faces get pinned:

  (a) `model_to_mjcf(material=..., actuators=...)` — the ContactMaterial
      presets and the position-servo layout are reachable from Python (they
      were engine-only: reachable from nowhere scriptable);
  (b) the asset doctor's Python severity spelling is "warning" — ONE spelling
      across the Python doctor surface (dataset doctor + trajectory lint
      already said "warning"; the asset doctor used to say "warn");
  (c) `caliper analyze/collide/reach --json` parse as JSON;
  (d) `caliper data delete/split/merge/tag` — dataset surgery from the CLI
      (the ops were Python/Studio-only), verified against `DatasetReaderV3`.

CLI tests follow the test_smoke.py convention: skip when the debug binary is
not built.
"""

import json
import pathlib
import subprocess

import pytest

import caliper

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = ROOT / "oracle" / "fixtures" / "robots"
CLI = ROOT / "target" / "debug" / "caliper"

needs_cli = pytest.mark.skipif(not CLI.exists(), reason="CLI binary not built")


def _robot(name="toy"):
    return caliper.Robot.from_urdf(str(FIX / f"{name}.urdf"))


def _run(*args):
    """Run the CLI, assert success, return stdout."""
    r = subprocess.run([str(CLI), *args], capture_output=True, text=True)
    assert r.returncode == 0, f"caliper {' '.join(args)} failed: {r.stderr}"
    return r.stdout


# ---------- (a) model_to_mjcf material= / actuators= ----------


def test_mjcf_material_preset_stamps_geoms():
    r = _robot("dyn_pendulum2")
    # No material (the default): byte-identical pre-material output — no
    # solver attributes anywhere.
    plain = caliper.model_to_mjcf(r, ground=0.0)
    assert "solref" not in plain and "solimp" not in plain

    # Preset by name: the documented Rubber knobs appear on emitted geoms
    # (the ground plane is always a geom, so this holds for any robot).
    xml = caliper.model_to_mjcf(r, ground=0.0, material="rubber")
    assert 'solref="0.01 1.0"' in xml
    assert 'friction="1.2 0.01 0.0002"' in xml
    # Case-insensitive preset names.
    assert caliper.model_to_mjcf(r, ground=0.0, material="Rubber") == xml


def test_mjcf_material_custom_dict_and_rejections():
    r = _robot("dyn_pendulum2")
    xml = caliper.model_to_mjcf(
        r,
        ground=0.0,
        material={
            "solref": (0.004, 1.0),
            "solimp": (0.9, 0.95, 0.001),
            "friction": (0.7, 0.005, 0.0001),
        },
    )
    assert 'solref="0.004 1.0"' in xml and 'solimp="0.9 0.95 0.001"' in xml

    with pytest.raises(ValueError, match="unknown material preset"):
        caliper.model_to_mjcf(r, material="granite")
    with pytest.raises(ValueError, match="missing"):
        caliper.model_to_mjcf(r, material={"solref": (0.01, 1.0)})
    with pytest.raises(ValueError, match="solref"):
        # Value validation is the generator's: negative timeconst rejected.
        caliper.model_to_mjcf(
            r,
            material={
                "solref": (-0.01, 1.0),
                "solimp": (0.9, 0.95, 0.001),
                "friction": (1.0, 0.005, 0.0001),
            },
        )


def test_mjcf_actuators_kwarg_emits_position_servos():
    r = _robot("dyn_pendulum2")
    plain = caliper.model_to_mjcf(r)
    assert "<actuator>" not in plain  # torque-direct default unchanged
    xml = caliper.model_to_mjcf(r, actuators=True, kp=250.0, kv=12.0)
    assert "<actuator>" in xml
    assert xml.count("<position") == r.ndof  # one servo per joint
    assert 'kp="250.0"' in xml and 'kv="12.0"' in xml


# ---------- (b) asset-doctor severity spelling ----------


def test_asset_doctor_says_warning_not_warn():
    rep = caliper.doctor(str(FIX / "doctor_visual_only.urdf"))
    sevs = {f["severity"] for f in rep["findings"]}
    assert "warn" not in sevs, "the pre-1.0 'warn' spelling is retired"
    assert "warning" in sevs  # A005 is a Warning-severity finding
    assert sevs <= {"error", "warning", "info"}


# ---------- (c) --json on analyze / collide / reach ----------


@needs_cli
def test_cli_analyze_json():
    v = json.loads(
        _run("analyze", str(FIX / "toy.urdf"), "--joints", "0.1,0.2", "--json")
    )
    assert v["robot"] == "toy2"
    assert v["kind"] in ("none", "wrist", "elbow", "boundary")
    assert isinstance(v["manipulability"], float) and len(v["sigma"]) == 3
    r = _robot()
    py = r.analyze([0.1, 0.2])
    assert abs(v["manipulability"] - py["manipulability"]) < 1e-12


@needs_cli
def test_cli_collide_json():
    v = json.loads(
        json_out := _run(
            "collide",
            str(FIX / "collide_arm.urdf"),
            "--joints",
            "0,0,0",
            "--ground",
            "-0.1",
            "--contacts",
            "--json",
        )
    )
    assert isinstance(v["collision"], bool)
    assert isinstance(v["self_pairs"], list) and isinstance(v["world_hits"], list)
    assert "contacts" in v  # present because --contacts was passed
    assert json_out.lstrip().startswith("{")  # pure JSON on stdout


@needs_cli
def test_cli_reach_json():
    v = json.loads(
        _run(
            "reach",
            str(FIX / "toy.urdf"),
            "--target",
            "1,0,0,0,1,0,0,0,1,0.1,0.0,0.1",
            "--json",
        )
    )
    assert v["status"] in ("reachable", "blocked", "unreachable")
    assert (v["q"] is None) or isinstance(v["q"], list)


# ---------- (d) caliper data delete / split / merge / tag ----------


def _record_v3(tmp_path, episodes=3, frames=10, fps=20):
    r = _robot()
    out = tmp_path / "ds"
    rec = caliper.RecorderV3(r, str(out), fps=fps)
    for e in range(episodes):
        rec.start_episode(f"task {e}")
        for k in range(frames):
            q = [0.01 * e + 0.001 * k, -0.002 * k]
            rec.append(q, q, k / fps)
        rec.finalize_episode()
    rec.close()
    return out


@needs_cli
def test_cli_data_edit_verbs(tmp_path):
    root = _record_v3(tmp_path)

    out = _run("data", "split", str(root), "--episode", "0", "--frame", "5")
    assert "SPLIT" in out
    assert caliper.DatasetReaderV3.open(str(root)).total_episodes == 4

    out = _run("data", "merge", str(root), "--first", "0", "--second", "1")
    assert "MERGE" in out
    assert caliper.DatasetReaderV3.open(str(root)).total_episodes == 3

    out = _run("data", "tag", str(root), "--episode", "1", "--add", "good,retry")
    assert "ep 1: good, retry" in out
    out = _run("data", "tag", str(root), "--episode", "1", "--remove", "retry")
    assert "retry" not in out and "good" in out
    assert caliper.dataset_read_tags(str(root)) == {1: ["good"]}

    out = _run("data", "delete", str(root), "--episodes", "0")
    assert "1 episode(s)" in out
    rd = caliper.DatasetReaderV3.open(str(root))
    assert rd.total_episodes == 2
    # tags were remapped by the edit engine: old ep 1 is now ep 0
    assert caliper.dataset_read_tags(str(root)) == {0: ["good"]}

    # tag listing with no flags is read-only
    out = _run("data", "tag", str(root))
    assert "ep 0: good" in out
