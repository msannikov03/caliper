"""Sim-camera oracle: MuJoCo offscreen frames -> image-conditioned v3.0 datasets.

Proves the sim camera collector (`caliper_learn.sim_camera` + `collect_sim`)
end-to-end against REAL upstream consumers, no mocks:

  (a) `collect_camera_dataset` (planner-labelled demos + per-frame MuJoCo
      offscreen renders through `caliper.RecorderV3(image_features=...)`)
      produces a dataset the real `LeRobotDataset` loads DIRECTLY: image
      tensors come back CHW float32 in [0, 1] (lerobot's PIL-decode /255
      convention), state/action/task values intact vs `DatasetReaderV3`;
  (b) an image-conditioned tiny ACT (lerobot's own ACTPolicy, resnet18,
      dim_model=64) takes two optimizer steps on a real chunked DataLoader
      batch — loss finite and decreasing;
  (c) determinism: recollecting with the same seed yields byte-identical PNG
      frames (MuJoCo offscreen pixels were verified byte-stable across
      processes; the planner and PIL encoder are deterministic too).

Offline notes as in test_dataset_v3.py: HF_HUB_OFFLINE=1 precedes the lerobot
import; ACT uses `pretrained_backbone_weights=None` so nothing is downloaded.
Skips (never fakes a pass): numpy/torch/mujoco/PIL/lerobot missing, or caliper
built without the sim-camera bindings (rebuild via maturin develop).
"""

import io
import os
import pathlib

import pytest

import caliper

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must precede lerobot import

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco", reason="mujoco not installed")
pytest.importorskip("PIL", reason="pillow not installed")
pytest.importorskip("lerobot", reason="lerobot not installed")
lerobot_dataset_mod = pytest.importorskip("lerobot.datasets.lerobot_dataset")
LeRobotDataset = lerobot_dataset_mod.LeRobotDataset

pytestmark = pytest.mark.skipif(
    not all(
        hasattr(caliper, c)
        for c in ("RecorderV3", "DatasetReaderV3", "model_to_mjcf",
                  "CollisionModel", "Planner")
    ),
    reason="caliper lacks sim-camera bindings — rebuild (maturin develop)",
)

pytest.importorskip(
    "caliper_learn", reason="learn sidecar not installed (pip install -e learn)"
)
from caliper_learn.collect_sim import DEFAULT_IMAGE_KEY, collect_camera_dataset
from caliper_learn.sim_camera import SimCameraScene

ROOT = pathlib.Path(__file__).resolve().parents[2]
URDF = str(ROOT / "oracle" / "fixtures" / "robots" / "collide_arm.urdf")

REPO_ID = "caliper/sim_cam"
FPS = 30
H = W = 96
SEED = 7
N_EPISODES = 2
MAX_FRAMES = 30
CHUNK = 10  # ACT action-chunk length


def _collect(root) -> str:
    return collect_camera_dataset(
        root, n_episodes=N_EPISODES, urdf=URDF, fps=FPS, seed0=SEED,
        width=W, height=H, max_frames=MAX_FRAMES,
    )


@pytest.fixture(scope="module")
def ds_root(tmp_path_factory) -> pathlib.Path:
    """Collect ONCE: 2 episodes x <=30 frames of 96x96 camera demos."""
    return pathlib.Path(_collect(tmp_path_factory.mktemp("sim_cam") / REPO_ID))


@pytest.fixture(scope="module")
def lr(ds_root):
    """The dataset loaded by REAL lerobot, straight from disk."""
    return LeRobotDataset(REPO_ID, root=str(ds_root))


# ---------------------------------------------------------------- the scene


def test_scene_renders_the_robot():
    """SimCameraScene contract: HWC uint8 RGB, robot visible, pose-dependent
    pixels, PNG bytes decode back exactly, depth is metric float32."""
    Image = pytest.importorskip("PIL.Image")
    robot = caliper.Robot.from_urdf(URDF)
    with SimCameraScene.from_robot(robot, width=W, height=H, ground=-0.1) as sc:
        q0 = [0.0] * robot.ndof
        q1 = [0.4] * robot.ndof
        rgb = sc.render(q0)
        assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
        assert int((rgb.sum(axis=2) > 0).sum()) > 0  # lit scene, not all-black
        assert not np.array_equal(rgb, sc.render(q1))  # the ARM is in frame

        png = sc.png(q0)
        assert png.startswith(b"\x89PNG")
        back = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        assert np.array_equal(back, rgb)  # lossless round-trip

        depth = sc.render_depth(q0)
        assert depth.shape == (H, W) and depth.dtype == np.float32
        assert np.isfinite(depth).all() and float(depth.min()) > 0.0
        # depth toggle must not poison the next RGB render
        assert sc.render(q0).shape == (H, W, 3)


# ------------------------------------------------- (a) real lerobot loads it


def test_lerobot_loads_camera_dataset(lr):
    assert lr.meta.total_episodes == N_EPISODES
    assert int(lr.fps) == FPS
    feat = lr.meta.features[DEFAULT_IMAGE_KEY]
    assert feat["dtype"] == "image"
    assert list(feat["shape"]) == [H, W, 3]
    assert list(feat["names"]) == ["height", "width", "channels"]

    img = lr[0][DEFAULT_IMAGE_KEY]
    # lerobot image convention: PIL decode + /255 -> CHW float32 in [0, 1]
    assert tuple(img.shape) == (3, H, W)
    assert img.dtype == torch.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert float(img.max()) > 0.0  # not a black frame


def test_state_action_and_frames_intact(ds_root, lr):
    """Per-frame parity: lerobot values == DatasetReaderV3 values == written;
    every frame carries a decodable 96x96 PNG of the right episode length."""
    from PIL import Image

    rd = caliper.DatasetReaderV3.open(str(ds_root))
    assert rd.total_episodes == N_EPISODES
    assert rd.image_features == [DEFAULT_IMAGE_KEY]
    ndof = rd.ndof

    start = 0  # episodes are stored contiguously in episode order
    for ep in range(N_EPISODES):
        states, actions, times = rd.read_episode(ep)
        assert 2 <= len(states) <= MAX_FRAMES
        pngs = rd.read_episode_images(ep)[DEFAULT_IMAGE_KEY]
        assert len(pngs) == len(states)
        for k in (0, len(states) - 1):  # first + terminal hold frame
            item = lr[start + k]
            assert item["episode_index"].item() == ep
            assert item["frame_index"].item() == k
            st = item["observation.state"]
            ac = item["action"]
            assert tuple(st.shape) == (ndof,) and st.dtype == torch.float32
            assert tuple(ac.shape) == (ndof,) and ac.dtype == torch.float32
            assert np.allclose(st.numpy(), states[k], atol=1e-5)
            assert np.allclose(ac.numpy(), actions[k], atol=1e-5)
            assert abs(item["timestamp"].item() - times[k]) < 1e-4
            # the parquet PNG decodes to exactly the tensor lerobot serves
            raw = np.asarray(Image.open(io.BytesIO(pngs[k])).convert("RGB"))
            assert raw.shape == (H, W, 3)
            served = (item[DEFAULT_IMAGE_KEY] * 255.0).round().to(torch.uint8)
            assert np.array_equal(served.permute(1, 2, 0).numpy(), raw)
        # terminal frame is the hold-at-goal sample: state == action
        assert np.allclose(states[-1], actions[-1], atol=1e-6)
        start += len(states)


# ----------------------------------- (b) image-conditioned ACT training step


def test_tiny_act_trains_on_camera_batch(ds_root, lr):
    """Two optimizer steps of lerobot's OWN image-conditioned ACT on a real
    chunked batch from the collected dataset: loss finite and decreasing."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    ndof = caliper.DatasetReaderV3.open(str(ds_root)).ndof
    # chunked view: action -> (CHUNK, ndof) + action_is_pad, lerobot-native
    ds = LeRobotDataset(
        REPO_ID, root=str(ds_root),
        delta_timestamps={"action": [i / FPS for i in range(CHUNK)]},
    )
    torch.manual_seed(0)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)))
    assert batch["action"].shape == (4, CHUNK, ndof)
    assert batch["action_is_pad"].shape == (4, CHUNK)
    assert batch[DEFAULT_IMAGE_KEY].shape == (4, 3, H, W)

    # normalization stats straight from the WRITER's stats.json via lerobot
    stats = {}
    for key in ("observation.state", DEFAULT_IMAGE_KEY, "action"):
        stats[key] = {
            s: torch.as_tensor(np.asarray(lr.meta.stats[key][s]), dtype=torch.float32)
            for s in ("mean", "std")
        }
    assert tuple(stats[DEFAULT_IMAGE_KEY]["mean"].shape) == (3, 1, 1)
    assert tuple(stats[DEFAULT_IMAGE_KEY]["std"].shape) == (3, 1, 1)
    assert float(stats[DEFAULT_IMAGE_KEY]["mean"].min()) >= 0.0
    assert float(stats[DEFAULT_IMAGE_KEY]["mean"].max()) <= 1.0
    for key, st in stats.items():
        for s, v in st.items():
            assert torch.isfinite(v).all(), f"non-finite {key}.{s}"
        st["std"] = st["std"].clamp_min(1e-4)  # guard /0 on constant dims

    cfg = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(ndof,)),
            DEFAULT_IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ndof,))},
        chunk_size=CHUNK, n_action_steps=CHUNK,
        dim_model=64, n_heads=2, dim_feedforward=128,
        n_encoder_layers=2, n_decoder_layers=1,
        vision_backbone="resnet18", pretrained_backbone_weights=None,
        use_vae=False, device="cpu",
    )
    pol = ACTPolicy(cfg, dataset_stats=stats)
    pol.train()
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)

    losses = []
    for _ in range(2):
        loss, _out = pol.forward(batch)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[1] < losses[0], f"loss did not decrease: {losses}"

    # inference contract: one action from observation.* keys only
    pol.eval()
    pol.reset()
    with torch.no_grad():
        a = pol.select_action(
            {k: v[:1] for k, v in batch.items() if k.startswith("observation")}
        )
    assert tuple(a.shape) == (1, ndof)


# --------------------------------------------------------- (c) determinism


def test_same_seed_identical_png_bytes(ds_root, tmp_path):
    """Recollect with the same seed -> byte-identical camera frames (pixel
    determinism was verified across processes during recon; the planner and
    the PIL encoder are deterministic, so the whole pipeline is)."""
    other = _collect(tmp_path / REPO_ID)
    a = caliper.DatasetReaderV3.open(str(ds_root))
    b = caliper.DatasetReaderV3.open(str(other))
    assert b.total_episodes == a.total_episodes
    for ep in range(a.total_episodes):
        pa = a.read_episode_images(ep)[DEFAULT_IMAGE_KEY]
        pb = b.read_episode_images(ep)[DEFAULT_IMAGE_KEY]
        assert pa[0] == pb[0]  # frame 0: the required byte-identity probe
        assert pa == pb  # and in fact every frame of the episode
        sa, aa, ta = a.read_episode(ep)
        sb, ab, tb = b.read_episode(ep)
        assert sa == sb and aa == ab and ta == tb
