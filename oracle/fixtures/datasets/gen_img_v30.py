"""Regenerate `img_v30/` — a tiny LeRobotDataset v3.0 WITH an image feature,
produced by lerobot 0.4.4's OWN writer (`LeRobotDataset.create` + `add_frame`
+ `save_episode` + `finalize`, `use_videos=False`). It is the hermetic ground
truth for `caliper-dataset`'s image-column tests
(`crates/caliper-dataset/tests/lerobot_v30_images.rs`): the data parquet
carries the `struct<bytes: binary, path: string>` PNG column exactly as
lerobot embeds it.

Frames are deterministic 32x32 RGB gradients (no RNG), so the fixture is
byte-stable across regenerations up to PIL's PNG encoder version.

Usage:
    cd <repo root>
    .venv/bin/python oracle/fixtures/datasets/gen_img_v30.py /tmp/imggen
    rm -rf oracle/fixtures/datasets/img_v30
    cp -R /tmp/imggen/img_v30 oracle/fixtures/datasets/img_v30
"""

import os
import shutil
import sys

import numpy as np

H, W = 32, 32
FPS = 30
EPISODES = 2
FRAMES = 3


def frame_pixels(ep: int, i: int) -> np.ndarray:
    """Deterministic RGB gradient, distinct per (episode, frame)."""
    y = np.arange(H, dtype=np.uint16)[:, None]
    x = np.arange(W, dtype=np.uint16)[None, :]
    r = (y * 8 + i * 10) % 256
    g = (x * 8 + ep * 40) % 256
    b = ((y + x) * 4 + ep * 20 + i * 5) % 256
    return np.stack(
        [np.broadcast_to(c, (H, W)) for c in (r, g, b)], axis=-1
    ).astype(np.uint8)


def main(workdir: str) -> None:
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = os.path.join(workdir, "lr_home")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = os.path.join(workdir, "img_v30")
    if os.path.exists(root):
        shutil.rmtree(root)
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["j1", "j2"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["j1", "j2"]},
        "observation.images.cam": {
            "dtype": "image",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
    }
    ds = LeRobotDataset.create(
        repo_id="caliper/img_v30",
        fps=FPS,
        features=features,
        root=root,
        robot_type="caliper",
        use_videos=False,
    )
    for ep in range(EPISODES):
        for i in range(FRAMES):
            ds.add_frame(
                {
                    "observation.state": np.array([0.1 * i, 0.2 * i], dtype=np.float32),
                    "action": np.array([0.3 * i, 0.4 * i], dtype=np.float32),
                    "observation.images.cam": frame_pixels(ep, i),
                    "task": "demo",
                }
            )
        ds.save_episode()
    ds.finalize()
    print(f"wrote {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/imggen")
