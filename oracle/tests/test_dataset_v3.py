"""v3.0-native oracle: Caliper's OWN LeRobotDataset v3.0 writer/reader vs real lerobot.

test_lerobot_roundtrip.py proves the v2.1 writer survives lerobot's official
v2.1->v3.0 CONVERTER; this file proves the new `caliper-dataset` crate speaks
v3.0 NATIVELY, in both directions and with NO converter in the loop:

  (a) caliper.RecorderV3 -> real LeRobotDataset (delta_timestamps) loads the
      dataset DIRECTLY: per-frame values, windowing, edge padding, task strings;
  (b) one verified-decreasing SGD step of a tiny torch policy on a DataLoader
      batch drawn from the natively-written dataset;
  (c) caliper.DatasetReaderV3 reads back exactly what caliper.RecorderV3 wrote;
  (d) cross-direction: a lerobot-WRITTEN v3.0 dataset (official converter run
      on a Caliper v2.1 recording) is read by caliper.DatasetReaderV3.

Offline notes as in test_lerobot_roundtrip.py: HF_HUB_OFFLINE=1 is set before
lerobot is imported so any accidental hub call raises instead of networking;
``LeRobotDataset(repo_id, root=...)`` loads straight from disk. Skips (never
fakes a pass): lerobot/torch/numpy missing, converter module missing (d only).
"""

import json
import os
import pathlib

import pytest

import caliper
from test_lerobot_dataset import _record

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must precede lerobot import

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("lerobot", reason="lerobot not installed")
lerobot_dataset_mod = pytest.importorskip("lerobot.datasets.lerobot_dataset")
LeRobotDataset = lerobot_dataset_mod.LeRobotDataset

pytestmark = pytest.mark.skipif(
    not all(
        hasattr(caliper, c)
        for c in ("ControlLoop", "RecorderV3", "DatasetReaderV3")
    ),
    reason="caliper lacks v3.0 dataset bindings — rebuild (maturin develop)",
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "oracle" / "fixtures" / "robots"

REPO_ID = "caliper/native_v3"
FPS = 50
DELTAS = [-0.1, 0.0]  # exactly -5 frames and "now" at 50 fps
SHIFT = round(-DELTAS[0] * FPS)
# 2 episodes, 2 distinct tasks: exercises multi-episode offsets + task mapping.
EPISODES = [
    ("reach a pose", [0.2, -0.1, 0.3, 0.0, 0.1, 0.0], 120),
    ("wave", [-0.1, 0.2, -0.2, 0.1, 0.0, 0.1], 80),
]


def _record_v3(ds_root: pathlib.Path):
    """Record EPISODES through the py-face v3.0 recorder; per-episode t from 0."""
    r = caliper.Robot.from_urdf(str(ROBOTS / "showcase6.urdf"))
    rec = caliper.RecorderV3(r, str(ds_root), FPS)
    recorded = []
    for task, goal, ticks in EPISODES:
        cl = caliper.ControlLoop(r, dt=1.0 / FPS)  # fresh loop: t starts at 0
        times, states, actions = cl.rollout_to(goal, ticks)
        rec.start_episode(task)
        for s, a, t in zip(states, actions, times):
            rec.append(s, a, t)
        rec.finalize_episode()
        recorded.append((task, states, actions, times))
    root = pathlib.Path(rec.close())
    assert root == ds_root
    return recorded


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    """Record once via RecorderV3, load once via real lerobot — NO converter."""
    ds_root = tmp_path_factory.mktemp("dataset_v3") / REPO_ID
    recorded = _record_v3(ds_root)
    ds = LeRobotDataset(
        REPO_ID,
        root=str(ds_root),
        delta_timestamps={"observation.state": DELTAS},
    )
    return ds, ds_root, recorded


def test_native_layout_is_v30_on_disk(native):
    _, ds_root, recorded = native
    info = json.loads((ds_root / "meta/info.json").read_text())
    total = sum(len(states) for _, states, _, _ in recorded)
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == len(EPISODES)
    assert info["total_frames"] == total
    assert info["total_tasks"] == 2
    assert (ds_root / "data/chunk-000/file-000.parquet").exists()
    assert (ds_root / "meta/episodes/chunk-000/file-000.parquet").exists()
    assert (ds_root / "meta/tasks.parquet").exists()
    assert (ds_root / "meta/stats.json").exists()
    # v3.0-native: no v2.1 sidecars, nothing to convert.
    assert not (ds_root / "meta/episodes.jsonl").exists()


def test_lerobot_loads_native_frames_windowing_padding(native):
    ds, _, recorded = native
    ndof = len(recorded[0][1][0])
    n0 = len(recorded[0][1])
    assert len(ds) == sum(len(states) for _, states, _, _ in recorded)
    assert int(ds.fps) == FPS
    assert ds.meta.total_episodes == len(EPISODES)

    # (episode, local index) probes: starts, the padding edge, interior, ends.
    probes = [
        (0, 0), (0, SHIFT - 1), (0, SHIFT), (0, 42), (0, n0 - 1),
        (1, 0), (1, SHIFT), (1, len(recorded[1][1]) - 1),
    ]
    for ep, i in probes:
        task, states, actions, times = recorded[ep]
        start = 0 if ep == 0 else n0
        item = ds[start + i]
        # --- windowed state: row 1 = frame i, row 0 = frame i-SHIFT clamped
        # to THIS episode's start (padding never leaks across episodes) ---
        win = item["observation.state"]
        assert tuple(win.shape) == (2, ndof)
        assert win.dtype == torch.float32
        assert np.allclose(win[1].numpy(), states[i], atol=1e-5)
        past = max(i - SHIFT, 0)
        assert np.allclose(win[0].numpy(), states[past], atol=1e-5)
        pad = item["observation.state_is_pad"]
        assert pad.tolist() == [i < SHIFT, False]
        # --- scalars + action come back exactly as recorded ---
        assert np.allclose(item["action"].numpy(), actions[i], atol=1e-5)
        assert abs(item["timestamp"].item() - times[i]) < 1e-4
        assert item["frame_index"].item() == i
        assert item["episode_index"].item() == ep
        assert item["index"].item() == start + i
        assert item["task"] == task


def test_one_optimizer_step_on_native_batch(native):
    """A tiny BC policy takes one real gradient step on a DataLoader batch."""
    ds, _, recorded = native
    ndof = len(recorded[0][1][0])
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    batch = next(iter(loader))
    obs = batch["observation.state"][:, -1, :]  # current frame of the window
    act = batch["action"]
    assert obs.shape == (32, ndof) and act.shape == (32, ndof)

    model = torch.nn.Linear(ndof, ndof)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_before = torch.nn.functional.mse_loss(model(obs), act)
    loss_before.backward()
    opt.step()
    loss_after = torch.nn.functional.mse_loss(model(obs), act)
    assert torch.isfinite(loss_before) and torch.isfinite(loss_after)
    assert loss_after.item() < loss_before.item()


def test_readerv3_reads_back_writer_output(native):
    _, ds_root, recorded = native
    rd = caliper.DatasetReaderV3.open(str(ds_root))
    assert rd.total_episodes == len(EPISODES)
    assert rd.fps == FPS
    assert rd.ndof == len(recorded[0][1][0])
    assert rd.tasks == [t for t, _, _ in EPISODES]
    for ep, (task, states, actions, times) in enumerate(recorded):
        assert rd.episode_tasks(ep) == [task]
        rs, ra, rt = rd.read_episode(ep)
        assert len(rs) == len(states)
        for got, want in zip(rs, states):
            assert np.allclose(got, want, atol=1e-5)
        for got, want in zip(ra, actions):
            assert np.allclose(got, want, atol=1e-5)
        assert np.allclose(rt, times, atol=1e-4)


def test_readerv3_reads_lerobot_written_dataset(tmp_path):
    """Cross-direction: lerobot's OFFICIAL tooling writes, our reader reads."""
    convert_mod = pytest.importorskip(
        "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        reason="this lerobot has no v2.1->v3.0 converter (need lerobot >= 0.4)",
    )
    repo_id = "caliper/cross"
    ds_root = tmp_path / repo_id
    _, states, actions, times = _record(ds_root, fps=FPS, ticks=60)
    # Converts IN PLACE: v3.0 replaces ds_root, pristine v2.1 at `<name>_old`.
    convert_mod.convert_dataset(repo_id=repo_id, root=str(tmp_path), push_to_hub=False)

    rd = caliper.DatasetReaderV3.open(str(ds_root))
    assert rd.total_episodes == 1
    assert rd.fps == FPS
    assert rd.ndof == len(states[0])
    assert rd.tasks == ["reach a pose"]
    assert rd.episode_tasks(0) == ["reach a pose"]
    rs, ra, rt = rd.read_episode(0)
    assert len(rs) == len(states)
    for got, want in zip(rs, states):
        assert np.allclose(got, want, atol=1e-5)
    for got, want in zip(ra, actions):
        assert np.allclose(got, want, atol=1e-5)
    assert np.allclose(rt, times, atol=1e-4)

    # The v2.1 original next door is REJECTED (version gate, not a crash).
    with pytest.raises(ValueError, match="v3"):
        caliper.DatasetReaderV3.open(str(ds_root) + "_old")


def test_offline_edits_keep_dataset_lerobot_loadable(tmp_path):
    """Offline edit ops (delete middle episode, then split another) + the tags
    sidecar must leave a dataset that REAL lerobot still loads with correct
    lengths, boundaries, values and task strings — and the caliper_tags.json
    extension file under meta/ must be invisible to lerobot's loader."""
    edit_ops = ("dataset_delete_episodes", "dataset_split_episode",
                "dataset_read_tags", "dataset_write_tags")
    if not all(hasattr(caliper, f) for f in edit_ops):
        pytest.skip("caliper lacks dataset edit ops — rebuild (maturin develop)")

    repo_id = "caliper/edit_v3"
    ds_root = tmp_path / repo_id
    r = caliper.Robot.from_urdf(str(ROBOTS / "showcase6.urdf"))
    rec = caliper.RecorderV3(r, str(ds_root), FPS)
    plan = [
        ("hold zero", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 40),
        ("wiggle", [0.1, -0.1, 0.1, -0.1, 0.1, -0.1], 30),
        ("reach a pose", [0.2, -0.1, 0.3, 0.0, 0.1, 0.0], 50),
    ]
    recorded = []
    for task, goal, ticks in plan:
        cl = caliper.ControlLoop(r, dt=1.0 / FPS)
        times, states, actions = cl.rollout_to(goal, ticks)
        rec.start_episode(task)
        for s, a, t in zip(states, actions, times):
            rec.append(s, a, t)
        rec.finalize_episode()
        recorded.append((task, states))
    rec.close()

    caliper.dataset_write_tags(
        str(ds_root), {0: ["keep"], 1: ["bad-demo"], 2: ["success"]}
    )
    caliper.dataset_delete_episodes(str(ds_root), [1])   # drop "wiggle"
    caliper.dataset_split_episode(str(ds_root), 1, 20)   # old ep2: 50 → 20 + 30

    # Real lerobot loads the edited dataset (tags sidecar present in meta/).
    assert (ds_root / "meta/caliper_tags.json").exists()
    ds = LeRobotDataset(repo_id, root=str(ds_root))
    assert ds.meta.total_episodes == 3
    assert len(ds) == 40 + 20 + 30

    # Boundaries + renumbering: episode 1 = first half of the old "reach" ep.
    item = ds[40]
    assert item["episode_index"].item() == 1
    assert item["frame_index"].item() == 0
    assert item["task"] == "reach a pose"
    assert np.allclose(item["observation.state"].numpy(), recorded[2][1][0], atol=1e-5)
    # Episode 2 = second half; timestamps restart per episode after the split.
    item = ds[60]
    assert item["episode_index"].item() == 2
    assert item["frame_index"].item() == 0
    assert item["task"] == "reach a pose"
    assert abs(item["timestamp"].item()) < 1e-4
    assert np.allclose(item["observation.state"].numpy(), recorded[2][1][20], atol=1e-5)
    last = ds[89]
    assert last["episode_index"].item() == 2
    assert np.allclose(last["observation.state"].numpy(), recorded[2][1][49], atol=1e-5)

    # Task remap dropped the deleted-episode-only task; tags were remapped.
    rd = caliper.DatasetReaderV3.open(str(ds_root))
    assert rd.total_episodes == 3
    assert rd.tasks == ["hold zero", "reach a pose"]
    assert caliper.dataset_read_tags(str(ds_root)) == {
        0: ["keep"], 1: ["success"], 2: ["success"],
    }
