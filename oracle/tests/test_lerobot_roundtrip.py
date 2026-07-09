"""Phase-5 oracle: full ROUND-TRIP through REAL lerobot, not schema checks.

test_lerobot_dataset.py proves the on-disk v2.1 layout byte-by-byte; this file
proves a Caliper-recorded dataset SURVIVES the actual lerobot toolchain:

  Caliper Recorder (v2.1)
    -> official lerobot v2.1->v3.0 converter (offline, local root)
    -> real LeRobotDataset with delta_timestamps on observation.state
    -> per-frame values equal what we recorded (windowing + edge padding)
    -> one optimizer step of a tiny torch policy on a DataLoader batch.

Offline notes (verified on lerobot 0.4.4): ``convert_dataset(repo_id,
root=DIR, push_to_hub=False)`` operates fully locally when ``DIR/repo_id``
holds a v2.1 dataset — it validates ``codebase_version``, converts in place
(keeping the v2.1 original at ``<name>_old``), and never contacts the hub.
``LeRobotDataset(repo_id, root=...)`` likewise loads straight from disk. We
set HF_HUB_OFFLINE=1 before importing lerobot so any accidental hub call
raises instead of hitting the network.

Skips (never fakes a pass): lerobot missing, or lerobot without the v3.0
converter module (pre-0.4 releases shipped a different converter path).
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
convert_mod = pytest.importorskip(
    "lerobot.datasets.v30.convert_dataset_v21_to_v30",
    reason="this lerobot has no v2.1->v3.0 converter (need lerobot >= 0.4, e.g. 0.4.4)",
)
lerobot_dataset_mod = pytest.importorskip("lerobot.datasets.lerobot_dataset")
LeRobotDataset = lerobot_dataset_mod.LeRobotDataset

pytestmark = pytest.mark.skipif(
    not all(hasattr(caliper, c) for c in ("ControlLoop", "Recorder", "DatasetReader")),
    reason="caliper lacks Phase-5 dataset bindings — rebuild (maturin develop)",
)

REPO_ID = "caliper/roundtrip"
FPS = 50
TICKS = 200
DELTAS = [-0.1, 0.0]  # exactly -5 frames and "now" at 50 fps
SHIFT = round(-DELTAS[0] * FPS)


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    """Record once, convert once, load once — shared by every test below."""
    tmp = tmp_path_factory.mktemp("lerobot_roundtrip")
    ds_root = tmp / REPO_ID
    _, states, actions, times = _record(ds_root, fps=FPS, ticks=TICKS)
    # The converter resolves the dataset at Path(root)/repo_id and converts it
    # IN PLACE (v3.0 replaces ds_root; the v2.1 original moves to `<name>_old`).
    convert_mod.convert_dataset(repo_id=REPO_ID, root=str(tmp), push_to_hub=False)
    ds = LeRobotDataset(
        REPO_ID,
        root=str(ds_root),
        delta_timestamps={"observation.state": DELTAS},
    )
    return ds, ds_root, states, actions, times


def test_converter_produced_v30_layout(roundtrip):
    _, ds_root, states, _, _ = roundtrip
    info = json.loads((ds_root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == len(states)
    assert (ds_root / "data/chunk-000/file-000.parquet").exists()
    assert (ds_root / "meta/episodes/chunk-000/file-000.parquet").exists()
    assert (ds_root / "meta/tasks.parquet").exists()
    assert (ds_root / "meta/stats.json").exists()
    # Converter quirk: the pristine v2.1 dataset survives next to the result.
    old = pathlib.Path(str(ds_root) + "_old")
    assert (old / "meta/info.json").exists()
    assert json.loads((old / "meta/info.json").read_text())["codebase_version"] == "v2.1"


def test_frames_match_recording(roundtrip):
    ds, _, states, actions, times = roundtrip
    ndof = len(states[0])
    assert len(ds) == TICKS
    assert int(ds.fps) == FPS
    assert ds.meta.total_episodes == 1
    assert ds.meta.total_frames == TICKS

    for i in (0, SHIFT - 1, SHIFT, 42, TICKS - 1):
        item = ds[i]
        # --- windowed state: row 1 = frame i, row 0 = frame i-SHIFT (clamped) ---
        win = item["observation.state"]
        assert tuple(win.shape) == (2, ndof)
        assert win.dtype == torch.float32
        assert np.allclose(win[1].numpy(), states[i], atol=1e-5)
        past = max(i - SHIFT, 0)  # lerobot clamps to the episode start and pads
        assert np.allclose(win[0].numpy(), states[past], atol=1e-5)
        pad = item["observation.state_is_pad"]
        assert pad.tolist() == [i < SHIFT, False]
        # --- scalars + action come back exactly as recorded ---
        assert np.allclose(item["action"].numpy(), actions[i], atol=1e-5)
        assert abs(item["timestamp"].item() - times[i]) < 1e-4
        assert item["frame_index"].item() == i
        assert item["episode_index"].item() == 0
        assert item["index"].item() == i
        assert item["task"] == "reach a pose"


def test_one_optimizer_step_on_loaded_batch(roundtrip):
    """A tiny BC policy takes one real gradient step on a DataLoader batch."""
    ds, _, states, _, _ = roundtrip
    ndof = len(states[0])
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
