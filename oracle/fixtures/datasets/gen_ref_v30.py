"""Regenerate the ref_v30 fixture: record a tiny dataset via caliper's v2.1
Recorder, then convert with lerobot's OFFICIAL v2.1->v3.0 converter (offline).

Usage: .venv/bin/python oracle/fixtures/datasets/gen_ref_v30.py <out_dir>
Result lands at <out_dir>/caliper/ref (the pristine v2.1 copy at .../ref_old).
See README.md in this directory for how to install it as the fixture.
"""

import os
import pathlib
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must precede lerobot import

import caliper  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ROBOTS = REPO_ROOT / "oracle" / "fixtures" / "robots"
REPO_ID = "caliper/ref"
FPS = 50

out = pathlib.Path(sys.argv[1])
ds_root = out / REPO_ID

robot = caliper.Robot.from_urdf(str(ROBOTS / "showcase6.urdf"))
loop = caliper.ControlLoop(robot, dt=1.0 / FPS)
rec = caliper.Recorder(robot, str(ds_root), FPS)

# 3 short episodes, 2 distinct tasks (order matters: fixed task_index mapping).
for goal, task, ticks in [
    ([0.2, -0.1, 0.3, 0.0, 0.1, 0.0], "reach a pose", 25),
    ([-0.1, 0.2, -0.2, 0.1, 0.0, 0.1], "reach a pose", 30),
    ([0.0, 0.3, 0.1, -0.1, 0.2, -0.1], "wave", 20),
]:
    times, states, actions = loop.rollout_to(goal, ticks)
    rec.start_episode(task)
    for s, a, t in zip(states, actions, times):
        rec.append(s, a, t)
    rec.finalize_episode()
rec.close()

from lerobot.datasets.v30 import convert_dataset_v21_to_v30 as conv  # noqa: E402

conv.convert_dataset(repo_id=REPO_ID, root=str(out), push_to_hub=False)
print(f"v3.0 reference dataset at {ds_root}")
