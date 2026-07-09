"""Interop exporters — pure-Python bridges from Caliper artifacts to
third-party on-disk formats. No Rust involved; everything here reads what the
engine already wrote (or plain Python values) and re-emits it byte-for-byte in
the target project's own layout.

Two exporters:

  * `export_lerobot_calibration` — caliper-calib joint offsets → lerobot's
    per-robot-id calibration JSON (`{calibration_dir}/{robot_id}.json`).
  * `export_robomimic_hdf5` — a Caliper-recorded LeRobotDataset v2.1 root →
    a robomimic-style HDF5 file (`data/demo_N/{obs,actions,rewards,dones}`).

Heavy deps (`h5py`, `pyarrow`) are imported lazily inside the functions so
`import caliper` never pays for them and a missing extra fails with a clear
actionable message instead of an import-time crash.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
from typing import Any, Sequence

__all__ = ["export_lerobot_calibration", "export_robomimic_hdf5"]

# ---------------------------------------------------------------------------
# lerobot calibration JSON
# ---------------------------------------------------------------------------

# How the raw Homing_Offset register combines with the encoder reading differs
# per bus family. Verbatim from the installed lerobot (v0.4.4) sources:
#   lerobot/motors/feetech/feetech.py::_get_half_turn_homings
#       "Present_Position = Actual_Position - Homing_Offset"
#   lerobot/motors/dynamixel/dynamixel.py::_get_half_turn_homings
#       "Present_Position = Actual_Position + Homing_Offset"
# Caliper's `calibrate_joint_offsets` returns `delta` such that
# `true_angle = reported_q + delta`; solving each identity for the register
# value that makes the reported position equal the true angle gives the signs
# below (assuming a fresh calibration, i.e. no prior homing offset applied).
_BUS_HOMING_SIGN = {"feetech": -1, "dynamixel": +1}


def export_lerobot_calibration(
    offsets: Sequence[float],
    motor_names: Sequence[str],
    path: str | os.PathLike[str] | None = None,
    *,
    resolution: int = 4096,
    bus: str = "feetech",
) -> dict[str, dict[str, int]]:
    """Map caliper-calib joint offsets into lerobot's calibration JSON.

    Target schema — one JSON object per robot id, motor name → the fields of
    `lerobot.motors.motors_bus.MotorCalibration` (a dataclass:
    `{id, drive_mode, homing_offset, range_min, range_max}`, all ints),
    written by `lerobot.robots.robot.Robot._save_calibration` via
    `draccus.dump(..., indent=4)` to `{calibration_dir}/{robot_id}.json`.
    (Schema read from the lerobot installed in this repo's `.venv`, v0.4.4.)

    `offsets` are the radian joint-zero corrections from
    `caliper.calibrate_joint_offsets(...)["offsets"]` (`true = reported + delta`).
    Each is converted to encoder ticks with lerobot's own tick<->angle scale
    (`resolution - 1` ticks per revolution, cf. its `DEGREES` normalization),
    then signed per the `bus` register convention (see `_BUS_HOMING_SIGN`).

    Fields Caliper cannot know are filled with the honest defaults:
    `drive_mode = 0` (no direction flip) and the full encoder range
    `[0, resolution - 1]` — replace them with `record_ranges_of_motion`
    output before trusting normalized reads on real hardware.

    If `path` is given the dict is also written there as indent-4 JSON
    (draccus emits plain nested JSON for `dict[str, MotorCalibration]`, so
    `json.dump` is byte-compatible). Returns the calibration dict either way.
    """
    if len(offsets) != len(motor_names):
        raise ValueError(
            f"offsets ({len(offsets)}) and motor_names ({len(motor_names)}) length mismatch"
        )
    if not motor_names:
        raise ValueError("motor_names must be non-empty")
    if len(set(motor_names)) != len(motor_names):
        raise ValueError(f"motor_names must be unique, got {list(motor_names)}")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    if bus not in _BUS_HOMING_SIGN:
        raise ValueError(f"bus must be one of {sorted(_BUS_HOMING_SIGN)}, got {bus!r}")
    for name, delta in zip(motor_names, offsets):
        if not math.isfinite(delta):
            raise ValueError(f"offset for {name!r} is not finite: {delta}")

    sign = _BUS_HOMING_SIGN[bus]
    max_res = resolution - 1
    calibration: dict[str, dict[str, int]] = {}
    for i, (name, delta) in enumerate(zip(motor_names, offsets)):
        calibration[name] = {
            "id": i + 1,
            "drive_mode": 0,
            "homing_offset": sign * round(delta / (2.0 * math.pi) * max_res),
            "range_min": 0,
            "range_max": max_res,
        }

    if path is not None:
        p = pathlib.Path(path)
        if p.parent != pathlib.Path("."):
            p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(calibration, f, indent=4)
    return calibration


# ---------------------------------------------------------------------------
# robomimic HDF5
# ---------------------------------------------------------------------------


def _require(module: str, extra_hint: str) -> Any:
    """Import `module` lazily with an actionable error if it is missing."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"{module} is required for this exporter but is not installed — {extra_hint}"
        ) from e


def export_robomimic_hdf5(
    dataset_root: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Convert a Caliper-recorded LeRobotDataset v2.1 root into robomimic HDF5.

    Input is the layout `caliper.Recorder` writes (see
    `crates/caliper-hal/src/recorder.rs`): `meta/info.json` advertising a
    `data_path` template (`data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`),
    one parquet per episode with `observation.state` / `action` /
    `timestamp` columns.

    Output is robomimic's dataset structure (per its documentation and
    `robomimic/utils/dataset.py` / `robomimic/scripts/conversion` sources):

        data/                       group
          attrs["total"]            total state-action samples across demos
          attrs["env_args"]         JSON str: {env_name, type, env_kwargs}
          demo_{i}/                 one group per episode, unpadded index
            attrs["num_samples"]    frames in this demo
            obs/<key>               (N, ...) per-modality observations
            actions                 (N, ndof) float32
            rewards                 (N,) float64
            dones                   (N,) int64

    Honest assumptions, since Caliper is not a registered robomimic env:
      * `env_args.type` is `2` (robomimic's `EnvType.GYM_TYPE`) as a generic
        placeholder; `env_name` is the recorded `robot_type`; `env_kwargs`
        carries `{fps}`. Playback inside robomimic needs a real env wrapper.
      * observations map to a single low-dim key `obs/state`
        (= `observation.state`); `obs/timestamp` is kept alongside so no
        recorded signal is dropped. No `next_obs`/`states` are emitted
        (train with `hdf5_load_next_obs=False`; there is no sim state).
      * Caliper records no reward, so `rewards` is all zeros and `dones`
        is zero except 1 on each episode's final frame.

    Requires `h5py` (not shipped with caliper — `pip install h5py`) and
    `pyarrow`. Returns `{"out_path", "demos", "total"}`.
    """
    h5py = _require("h5py", "install it with `pip install h5py`")
    np = _require("numpy", "install it with `pip install numpy`")
    pq = _require("pyarrow.parquet", "install it with `pip install pyarrow`")

    root = pathlib.Path(dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"not a LeRobotDataset root (no {info_path})")
    info = json.loads(info_path.read_text())
    version = info.get("codebase_version")
    if version != "v2.1":
        raise ValueError(f"expected a LeRobotDataset v2.1 root, got codebase_version={version!r}")
    total_episodes = int(info["total_episodes"])
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info["data_path"]

    out = pathlib.Path(out_path)
    if out.parent != pathlib.Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with h5py.File(out, "w") as f:
        data = f.create_group("data")
        for ep in range(total_episodes):
            rel = data_path.format(episode_chunk=ep // chunks_size, episode_index=ep)
            table = pq.read_table(root / rel)
            states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
            actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
            timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float32)
            n = states.shape[0]
            dones = np.zeros(n, dtype=np.int64)
            if n > 0:
                dones[-1] = 1

            demo = data.create_group(f"demo_{ep}")
            demo.attrs["num_samples"] = n
            obs = demo.create_group("obs")
            obs.create_dataset("state", data=states)
            obs.create_dataset("timestamp", data=timestamps)
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("rewards", data=np.zeros(n, dtype=np.float64))
            demo.create_dataset("dones", data=dones)
            total += n

        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": info.get("robot_type", "caliper"),
                "type": 2,  # robomimic EnvType.GYM_TYPE — placeholder, see docstring
                "env_kwargs": {"fps": info.get("fps")},
            }
        )

    return {"out_path": str(out), "demos": total_episodes, "total": total}
