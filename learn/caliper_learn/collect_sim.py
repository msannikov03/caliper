"""Collect image-conditioned sim demonstrations: the planner-label machinery
from `collect.py` (one-step-lookahead frames, state = q_k, action = q_{k+1},
terminal hold-at-goal frame) PLUS a MuJoCo offscreen camera per frame, written
as a native LeRobotDataset v3.0 through `caliper.RecorderV3` with a
`dtype: "image"` feature (pre-encoded PNG bytes, no video/ffmpeg).

Deterministic given `seed0`: the planner and start/goal sampling are seeded
(same scheme as `collect.collect_demos`), MuJoCo offscreen pixels are
byte-identical across runs/processes, and the PIL PNG encoder is
deterministic — so reruns produce identical image bytes.
"""

from __future__ import annotations

import os

import numpy as np

from .collect import _DEFAULT_BOXES, _bounds, _resolve_urdf, _sample_free
from .sim_camera import SimCameraScene

DEFAULT_IMAGE_KEY = "observation.images.cam"


def collect_camera_dataset(
    out_dir: str | os.PathLike,
    *,
    n_episodes: int = 2,
    urdf: str | None = None,
    fps: int = 30,
    seed0: int = 0,
    width: int = 96,
    height: int = 96,
    image_key: str = DEFAULT_IMAGE_KEY,
    ground: float = -0.1,
    boxes=None,
    dt: float | None = None,
    max_frames: int | None = None,
    max_goal_tries: int = 200,
    task_template: str = "reach pose {ep}",
    scene: SimCameraScene | None = None,
) -> str:
    """Record `n_episodes` camera demonstrations into a LeRobotDataset v3.0.

    Mirrors `collect.collect_demos(backend="planner")` episode-for-episode:
    rejection-sample a collision-free start+goal, plan a retimed trajectory,
    store one-step-lookahead frames plus a terminal hold frame — and render
    each frame's STATE q_k through `scene` (built from the robot's own URDF
    via `model_to_mjcf` when not supplied). `dt` defaults to `1/fps` so stored
    timestamps match the plan's physical spacing. `max_frames` caps each
    episode's length (trajectory truncation; the terminal hold frame then
    settles at the truncated goal). Returns the dataset root path.

    NOTE: the default fixture (`collide_arm`) has inertials, so `model_to_mjcf`
    accepts it; a custom `urdf` without `<inertial>` data needs an explicit
    pre-built `scene`.
    """
    import caliper  # runtime dep (built via maturin), not a packaging dep

    if max_frames is not None and max_frames < 2:
        raise ValueError(f"max_frames must be >= 2, got {max_frames}")
    boxes = _DEFAULT_BOXES if boxes is None else boxes
    dt = 1.0 / fps if dt is None else dt
    robot = caliper.Robot.from_urdf(_resolve_urdf("planner", urdf))
    own_scene = scene is None
    if own_scene:
        scene = SimCameraScene.from_robot(robot, width=width, height=height, ground=ground)

    rec = caliper.RecorderV3(
        robot, str(out_dir), fps=fps,
        image_features=[(image_key, scene.height, scene.width, 3)],
    )
    cm = caliper.CollisionModel(robot, ground=ground, boxes=boxes, margin=0.0)
    bounds = _bounds(robot)
    try:
        for ep in range(n_episodes):
            rng = np.random.default_rng(seed0 + ep)
            start = _sample_free(rng, bounds, cm, max_goal_tries)
            goal = _sample_free(rng, bounds, cm, max_goal_tries)
            planner = caliper.Planner(robot, ground=ground, boxes=boxes, seed=seed0 + ep)
            _ts, qs, _qds = planner.plan_trajectory(start, goal, dt=dt)
            if len(qs) < 2:
                continue  # degenerate (start==goal); skip, as collect_demos does
            if max_frames is not None:
                qs = qs[:max_frames]  # cap: len(qs)-1 lookahead pairs + 1 hold
            rec.start_episode(task_template.format(ep=ep))
            for k in range(len(qs) - 1):  # state=q_k, action=q_{k+1}
                rec.append(qs[k], qs[k + 1], k / fps,
                           images={image_key: scene.png(qs[k])})
            last = len(qs) - 1  # terminal hold-at-goal frame (see collect.py)
            rec.append(qs[last], qs[last], last / fps,
                       images={image_key: scene.png(qs[last])})
            rec.finalize_episode()
        return rec.close()
    finally:
        if own_scene:
            scene.close()


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Collect Caliper sim camera demos -> LeRobotDataset v3.0 (image dtype)"
    )
    p.add_argument("out", help="output dataset directory")
    p.add_argument("-n", "--episodes", type=int, default=2)
    p.add_argument("--urdf", default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--height", type=int, default=96)
    p.add_argument("--max-frames", type=int, default=None)
    a = p.parse_args(argv)
    root = collect_camera_dataset(
        a.out, n_episodes=a.episodes, urdf=a.urdf, fps=a.fps, seed0=a.seed,
        width=a.width, height=a.height, max_frames=a.max_frames,
    )
    print(root)


if __name__ == "__main__":
    main()
