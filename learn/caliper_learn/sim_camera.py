"""Sim camera: MuJoCo offscreen rendering of a caliper Robot — the piece that
replaces a physical camera when collecting image-conditioned BC datasets.

`SimCameraScene` owns a `mujoco.MjModel`/`MjData` + offscreen `mujoco.Renderer`
built from `caliper.model_to_mjcf` plus a world-fixed over-the-shoulder
`<camera>` (injected through `model_to_mjcf(extra_xml=...)`, the documented
hook). `step(q)` sets qpos, runs `mj_forward`, renders, and PNG-encodes.

Backend: none configured — MuJoCo's Renderer picks a working headless backend
on its own (CGL on macos-arm64, EGL/OSMesa per `MUJOCO_GL` elsewhere); setting
env vars programmatically was verified unnecessary. Pixels are deterministic
(byte-identical across processes for the same qpos).

Encoding parity: PIL PNG at `compress_level=6` — PIL is a lerobot dependency,
and 6 is the level lerobot's own dtype-"image" embed path produces in the
parquet (its async image writer uses 1; we match the stored-bytes default).

Heavy deps (mujoco, PIL, caliper) are imported lazily so `import caliper_learn`
stays cheap and torch/mujoco-free.
"""

from __future__ import annotations

import io

import numpy as np

DEFAULT_CAMERA = "ots"
# Over-the-shoulder 3/4 view from (+x, -y, above) looking ~35 deg down at the
# origin; xyaxes = camera right then up vector (MuJoCo orthonormalizes and
# looks along -(x cross y)). Verified by rendering an actual arm. pos is for a
# ~0.6 m-reach arm and is scaled by reach/0.6 in `camera_xml`.
_CAM_POS = (1.1, -1.1, 0.9)
_CAM_XYAXES = "0.707 0.707 0 -0.4 0.4 0.825"
_REF_REACH = 0.6
_MIN_REACH = 0.3
_LIGHT_XML = '<light pos="0 0 3" dir="0 0 -1" directional="true"/>'


def robot_reach(robot) -> float:
    """Rough reach (m): distance from base to every frame origin at q = 0,
    maxed, floored at `_MIN_REACH` — only used to scale the default camera."""
    q0 = [0.0] * robot.ndof
    best = 0.0
    for name in robot.frame_names():
        pose = robot.fk(q0, name)  # 4x4 ROW-major
        best = max(best, float(np.hypot(np.hypot(pose[0][3], pose[1][3]), pose[2][3])))
    return max(best, _MIN_REACH)


def camera_xml(reach: float = _REF_REACH, name: str = DEFAULT_CAMERA) -> str:
    """World-fixed `<camera>` + `<light>` MJCF snippet, scaled to `reach`."""
    if not (np.isfinite(reach) and reach > 0.0):
        raise ValueError(f"reach must be finite and > 0, got {reach}")
    s = reach / _REF_REACH
    pos = " ".join(f"{c * s:.6g}" for c in _CAM_POS)
    return f'{_LIGHT_XML}<camera name="{name}" pos="{pos}" xyaxes="{_CAM_XYAXES}"/>'


class SimCameraScene:
    """Offscreen MuJoCo render of a caliper robot at arbitrary joint configs.

    Build with `SimCameraScene.from_robot(robot, ...)` (the normal path) or
    directly from a full MJCF string that already contains a named camera.
    Use as a context manager or call `close()` to free the GL context.
    """

    def __init__(self, mjcf_xml: str, *, width: int = 96, height: int = 96,
                 camera: str = DEFAULT_CAMERA, compress_level: int = 6):
        import mujoco  # lazy: keep caliper_learn importable without mujoco

        self.width = int(width)
        self.height = int(height)
        self.camera = camera
        self.compress_level = int(compress_level)
        self.model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        self._mujoco = mujoco

    @classmethod
    def from_robot(cls, robot, *, width: int = 96, height: int = 96,
                   camera: str = DEFAULT_CAMERA, ground: float | None = None,
                   extra_xml: str = "", reach: float | None = None,
                   compress_level: int = 6) -> "SimCameraScene":
        """Build from a caliper Robot via `caliper.model_to_mjcf`.

        `extra_xml` is appended verbatim inside `<worldbody>` AFTER the camera
        — the hook for prop `<geom>`s (targets, tables, obstacles). The default
        camera is over-the-shoulder, auto-scaled to the robot's reach (override
        with `reach`, or supply your own `<camera>` in `extra_xml` + `camera=`).
        """
        import caliper  # lazy runtime dep (built via maturin)

        r = robot_reach(robot) if reach is None else float(reach)
        xml = caliper.model_to_mjcf(
            robot, ground=ground, extra_xml=camera_xml(r, camera) + extra_xml,
        )
        return cls(xml, width=width, height=height, camera=camera,
                   compress_level=compress_level)

    def render(self, q) -> np.ndarray:
        """Render joint config `q` → (H, W, 3) uint8 RGB."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.model.nq,):
            raise ValueError(f"q has shape {q.shape}, model expects ({self.model.nq},)")
        self.data.qpos[:] = q
        self._mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def png(self, q) -> bytes:
        """Render `q` and PNG-encode (PIL, lerobot-parity compress level)."""
        return self.encode_png(self.render(q))

    def encode_png(self, rgb: np.ndarray) -> bytes:
        from PIL import Image  # lazy; PIL is a lerobot dependency

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="png", compress_level=self.compress_level)
        return buf.getvalue()

    def render_depth(self, q) -> np.ndarray:
        """Render `q` → (H, W) float32 depth in meters (optional extra)."""
        self.render(q)  # sets qpos + forward
        self._renderer.enable_depth_rendering()
        try:
            self._renderer.update_scene(self.data, camera=self.camera)
            return self._renderer.render()
        finally:
            self._renderer.disable_depth_rendering()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "SimCameraScene":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
