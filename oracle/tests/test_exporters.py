"""Oracle for the pure-Python interop exporters (`caliper/interop.py`).

Two exporters, two validation strategies — honest about what each proves:
  (1) lerobot calibration JSON: the per-motor field set is validated against
      the INSTALLED lerobot's `MotorCalibration` dataclass when importable
      (plus a draccus round-trip mirroring `Robot._load_calibration`), and
      against the literal v0.4.4 field set otherwise — so the test never
      silently weakens to "matches what we hardcoded" while lerobot is there.
  (2) robomimic HDF5: record a real episode through the control loop, export,
      reopen with h5py and check structure + shapes + values round-trip.
      SKIPPED (not failed) when h5py is absent — it is not a caliper
      dependency (`pip install h5py` to run it); robomimic itself is not
      importable here, so layout conformance rests on the documented
      structure, not a live `SequenceDataset` load.
"""

import json
import math
import pathlib

import pytest

import caliper

np = pytest.importorskip("numpy")

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"

pytestmark = pytest.mark.skipif(
    not all(hasattr(caliper, c) for c in ("export_lerobot_calibration", "export_robomimic_hdf5")),
    reason="caliper lacks the interop exporters — rebuild (maturin develop)",
)

# lerobot.motors.motors_bus.MotorCalibration fields as of lerobot v0.4.4 —
# the fallback oracle when lerobot itself is not importable.
MOTOR_CALIBRATION_FIELDS = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}


def _lerobot_calibration_fields():
    """Field set straight from the installed lerobot, or the literal fallback."""
    try:
        import dataclasses

        from lerobot.motors.motors_bus import MotorCalibration
    except ImportError:
        return MOTOR_CALIBRATION_FIELDS, False
    return {f.name for f in dataclasses.fields(MotorCalibration)}, True


# ---------------------------------------------------------------------------
# lerobot calibration JSON
# ---------------------------------------------------------------------------


def test_calibration_schema_and_values():
    offsets = [0.1, -0.02, 0.0]
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
    cal = caliper.export_lerobot_calibration(offsets, names)

    assert list(cal) == names
    fields, from_lerobot = _lerobot_calibration_fields()
    for name, entry in cal.items():
        assert set(entry) == fields, f"{name}: schema drift vs lerobot ({from_lerobot=})"
        assert all(isinstance(v, int) for v in entry.values()), name

    # ids are 1-based and sequential; defaults cover the full 4096 encoder.
    assert [cal[n]["id"] for n in names] == [1, 2, 3]
    assert all(cal[n]["drive_mode"] == 0 for n in names)
    assert all((cal[n]["range_min"], cal[n]["range_max"]) == (0, 4095) for n in names)

    # tick conversion uses lerobot's own scale: (resolution-1) ticks per turn.
    assert cal["shoulder_pan"]["homing_offset"] == -round(0.1 / (2 * math.pi) * 4095)
    assert cal["elbow_flex"]["homing_offset"] == 0


def test_calibration_bus_sign_conventions():
    """Feetech (Present = Actual - Homing) and Dynamixel (Present = Actual +
    Homing) need OPPOSITE register signs for the same radian offset."""
    fee = caliper.export_lerobot_calibration([0.3], ["m"], bus="feetech")
    dyn = caliper.export_lerobot_calibration([0.3], ["m"], bus="dynamixel")
    assert dyn["m"]["homing_offset"] == -fee["m"]["homing_offset"] != 0


def test_calibration_json_file_matches_lerobot_loader(tmp_path):
    offsets = [0.05, -0.1]
    names = ["waist", "wrist_roll"]
    path = tmp_path / "calib" / "my_robot.json"
    cal = caliper.export_lerobot_calibration(offsets, names, path, resolution=1024)

    # On-disk JSON == the returned dict, byte-compatible with draccus dump.
    assert json.loads(path.read_text()) == cal

    # Strongest check available: load the file exactly the way
    # lerobot.robots.robot.Robot._load_calibration does.
    try:
        import draccus
        from lerobot.motors.motors_bus import MotorCalibration
    except ImportError:
        pytest.skip("lerobot/draccus not installed — dataclass round-trip not verifiable")
    with open(path) as f, draccus.config_type("json"):
        loaded = draccus.load(dict[str, MotorCalibration], f)
    assert list(loaded) == names
    for name in names:
        assert loaded[name].homing_offset == cal[name]["homing_offset"]
        assert loaded[name].range_max == 1023


def test_calibration_rejects_bad_input():
    with pytest.raises(ValueError, match="length mismatch"):
        caliper.export_lerobot_calibration([0.1], ["a", "b"])
    with pytest.raises(ValueError, match="non-empty"):
        caliper.export_lerobot_calibration([], [])
    with pytest.raises(ValueError, match="unique"):
        caliper.export_lerobot_calibration([0.1, 0.2], ["a", "a"])
    with pytest.raises(ValueError, match="bus"):
        caliper.export_lerobot_calibration([0.1], ["a"], bus="canbus")
    with pytest.raises(ValueError, match="finite"):
        caliper.export_lerobot_calibration([float("nan")], ["a"])


# ---------------------------------------------------------------------------
# robomimic HDF5
# ---------------------------------------------------------------------------

needs_recorder = pytest.mark.skipif(
    not all(hasattr(caliper, c) for c in ("ControlLoop", "Recorder")),
    reason="caliper lacks Phase-5 dataset bindings — rebuild (maturin develop)",
)


def _record(out: pathlib.Path, fps=50, ticks=60):
    """Same recipe as test_lerobot_dataset: one real control-loop episode."""
    r = caliper.Robot.from_urdf(str(ROBOTS / "showcase6.urdf"))
    cl = caliper.ControlLoop(r, dt=1.0 / fps)
    goal = [0.2, -0.1, 0.3, 0.0, 0.1, 0.0]
    times, states, actions = cl.rollout_to(goal, ticks)
    rec = caliper.Recorder(r, str(out), fps)
    rec.start_episode("reach a pose")
    for s, a, t in zip(states, actions, times):
        rec.append(s, a, t)
    rec.finalize_episode()
    return rec.close(), states, actions


@needs_recorder
def test_robomimic_export_roundtrips_shapes(tmp_path):
    h5py = pytest.importorskip("h5py", reason="h5py not installed (pip install h5py)")
    root, states, actions = _record(tmp_path / "ds")
    out = tmp_path / "demo.hdf5"
    summary = caliper.export_robomimic_hdf5(root, out)
    n, ndof = len(states), len(states[0])
    assert summary == {"out_path": str(out), "demos": 1, "total": n}

    with h5py.File(out, "r") as f:
        data = f["data"]
        assert int(data.attrs["total"]) == n
        env_args = json.loads(data.attrs["env_args"])
        assert set(env_args) == {"env_name", "type", "env_kwargs"}
        assert env_args["type"] == 2  # robomimic EnvType.GYM_TYPE placeholder

        assert list(data.keys()) == ["demo_0"]
        demo = data["demo_0"]
        assert int(demo.attrs["num_samples"]) == n
        assert demo["obs/state"].shape == (n, ndof)
        assert demo["obs/timestamp"].shape == (n,)
        assert demo["actions"].shape == (n, ndof)
        assert np.allclose(demo["obs/state"][()], np.asarray(states), atol=1e-5)
        assert np.allclose(demo["actions"][()], np.asarray(actions), atol=1e-5)
        assert np.array_equal(demo["dones"][()], [0] * (n - 1) + [1])
        assert not demo["rewards"][()].any()
        assert "next_obs" not in demo and "states" not in demo


@needs_recorder
def test_robomimic_export_multi_episode(tmp_path):
    h5py = pytest.importorskip("h5py", reason="h5py not installed (pip install h5py)")
    out_ds = tmp_path / "ds"
    r = caliper.Robot.from_urdf(str(ROBOTS / "showcase6.urdf"))
    cl = caliper.ControlLoop(r, dt=0.02)
    rec = caliper.Recorder(r, str(out_ds), 50)
    lengths = (40, 25)
    for k, ticks in enumerate(lengths):
        times, states, actions = cl.rollout_to([0.1 * (k + 1)] * 6, ticks)
        rec.start_episode(f"episode {k}")
        for s, a, t in zip(states, actions, times):
            rec.append(s, a, t)
        rec.finalize_episode()
    root = rec.close()

    summary = caliper.export_robomimic_hdf5(root, tmp_path / "demos.hdf5")
    assert summary["demos"] == 2
    with h5py.File(summary["out_path"], "r") as f:
        assert sorted(f["data"].keys()) == ["demo_0", "demo_1"]
        got = [int(f[f"data/demo_{k}"].attrs["num_samples"]) for k in range(2)]
        assert got == list(lengths)
        assert int(f["data"].attrs["total"]) == sum(lengths)


def test_robomimic_rejects_non_dataset(tmp_path):
    pytest.importorskip("h5py", reason="h5py not installed (pip install h5py)")
    with pytest.raises(FileNotFoundError, match="info.json"):
        caliper.export_robomimic_hdf5(tmp_path / "nope", tmp_path / "x.hdf5")
