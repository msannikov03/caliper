"""dtype-"video" storage tests: the encode chain, the per-episode mp4 layout,
the `attach_video_metadata` post-write bridge, and THE GATE — a dataset this
module wrote loads through REAL lerobot (`LeRobotDataset`) and hands back
decoded video frames (CHW float32) matching the originally stored pixels
within codec tolerance.

Tolerances are MEASURED, not guessed (this mac, pyav 15.1 / svt-av1 crf30
preset12 / libx264 crf30, torchcodec 0.10 decode): the smooth synthetic
frames below decode to mean-abs-diff ~0.013 (av1) / ~0.019 (h264) in [0, 1],
the 64x64 sim-camera renders to ~0.011 max — thresholds 0.06 / 0.05 keep a
~3x margin without ever passing on garbage (a black or shuffled decode is
>0.2). Encoded bytes are NOT bit-stable across runs (multi-threaded
encoders), so every comparison here is decoded-content, never file hashes.

Honest skips, never fakes: no encoder chain -> the encoding tests skip with
`available()`'s reason; lerobot missing -> the load gates skip (lerobot
imports pyav itself, so "lerobot importable" implies a decoder exists);
mujoco missing -> only the sim-camera gate skips. Validation-order tests
(frame shape/dtype/fps) run everywhere — `encode_episode_video` validates
input BEFORE probing encoders.
"""

import os
import pathlib

import numpy as np
import pytest

caliper = pytest.importorskip("caliper")

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must precede any lerobot import

from caliper_learn.collect import _resolve_urdf  # noqa: E402
from caliper_learn.video import (  # noqa: E402
    DEFAULT_VIDEO_PATH,
    VideoRecorder,
    attach_video_metadata,
    available,
    encode_episode_video,
)

_OK, _REASON = available()
needs_encoder = pytest.mark.skipif(not _OK, reason=f"no video encoder: {_REASON}")

KEY = "observation.images.cam"
FPS = 30
H = W = 64  # libsvtav1's own floor — smallest size both codecs accept


def _frames(n: int, h: int = H, w: int = W, seed: int = 0) -> np.ndarray:
    """n smooth, per-frame-distinct (h, w, 3) uint8 frames: moving gradients
    plus a bright wandering square. Deterministic; codec-friendly content so
    the measured tolerances above hold."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w]
    out = []
    for k in range(n):
        img = np.stack(
            [(ys * 3 + k * 7) % 256, (xs * 3 + k * 11) % 256, ((ys + xs) * 2 + k * 5) % 256],
            axis=-1,
        ).astype(np.float64)
        cy = int(h / 2 + h / 4 * np.sin(k / 3))
        cx = int(w / 2 + w / 4 * np.cos(k / 3))
        img[max(0, cy - 6) : cy + 6, max(0, cx - 6) : cx + 6] = rng.uniform(200, 255, 3)
        out.append(img.astype(np.uint8))
    return np.stack(out)


# ------------------------------------------------------------ available()


def test_available_contract():
    ok, reason = available()
    assert isinstance(ok, bool) and isinstance(reason, str) and reason


def test_available_rejects_unknown_codec():
    ok, reason = available("mjpeg")
    assert not ok and "unsupported codec" in reason


# --------------------------------------------- encode: validation (no encoder)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((0, H, W, 3), dtype=np.uint8),  # no frames
        np.zeros((2, H, W, 4), dtype=np.uint8),  # 4 channels
        np.zeros((2, H, W), dtype=np.uint8),  # missing channel dim
        np.zeros((2, 63, W, 3), dtype=np.uint8),  # odd height (yuv420p)
        np.zeros((2, H, 62 + 1, 3), dtype=np.uint8),  # odd width
        np.zeros((2, H, W, 3), dtype=np.float32),  # wrong dtype
    ],
    ids=["empty", "rgba", "no-channels", "odd-h", "odd-w", "float"],
)
def test_encode_rejects_bad_frames(tmp_path, bad):
    with pytest.raises(ValueError):
        encode_episode_video(bad, FPS, tmp_path / "v.mp4")


def test_encode_rejects_ragged_frame_list(tmp_path):
    ragged = [np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W * 2, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="shape"):
        encode_episode_video(ragged, FPS, tmp_path / "v.mp4")


def test_encode_rejects_nonpositive_fps(tmp_path):
    with pytest.raises(ValueError, match="fps"):
        encode_episode_video(_frames(2), 0, tmp_path / "v.mp4")


# ------------------------------------------------- encode: real round-trips


@needs_encoder
@pytest.mark.parametrize("codec", ["libsvtav1", "h264"])
def test_encode_decode_roundtrip(tmp_path, codec):
    """Encoded file decodes (pure pyav, no lerobot) back to the same frame
    COUNT and the same CONTENT within codec tolerance; the container carries
    the canonical codec name `feature_info` promises; the returned duration
    is exact n/fps."""
    ok, reason = available(codec)
    if not ok:
        pytest.skip(reason)
    av = pytest.importorskip("av")

    fr = _frames(12, seed=3)
    out = tmp_path / "ep.mp4"
    dur = encode_episode_video(fr, FPS, out, codec=codec)
    assert dur == 12 / FPS and out.is_file()

    with av.open(str(out)) as c:
        stream = c.streams.video[0]
        assert stream.codec.canonical_name == {"libsvtav1": "av1", "h264": "h264"}[codec]
        assert stream.pix_fmt == "yuv420p"
        dec = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    assert dec.shape == fr.shape
    err = np.abs(dec.astype(np.float64) - fr.astype(np.float64)).mean() / 255.0
    assert err < 0.06, f"decoded content diverged: mean abs diff {err:.4f}"


@needs_encoder
def test_encode_refuses_overwrite(tmp_path):
    """lerobot's encode silently SKIPS an existing file; here that would drop
    an episode's video, so it must be an error."""
    out = tmp_path / "ep.mp4"
    encode_episode_video(_frames(2), FPS, out)
    with pytest.raises(FileExistsError):
        encode_episode_video(_frames(2), FPS, out)


# ------------------------------------------------------------ VideoRecorder


@needs_encoder
@pytest.mark.parametrize(
    "kwargs",
    [
        {"video_key": ""},
        {"video_key": "a/b"},
        {"fps": 0},
        {"chunks_size": 0},
    ],
    ids=["empty-key", "slash-key", "zero-fps", "zero-chunks"],
)
def test_recorder_rejects_bad_construction(tmp_path, kwargs):
    full = {"video_key": KEY, "fps": FPS} | kwargs
    with pytest.raises(ValueError):
        VideoRecorder(tmp_path, full["video_key"], full["fps"],
                      chunks_size=full.get("chunks_size", 1000))


@needs_encoder
def test_recorder_locks_frame_shape_and_validates(tmp_path):
    r = VideoRecorder(tmp_path, KEY, FPS)
    r.append(_frames(1)[0])
    with pytest.raises(ValueError, match="locked shape"):
        r.append(np.zeros((H, W * 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="odd"):
        r.append(np.zeros((H - 1, W, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        r.append(np.zeros((H, W, 3), dtype=np.float64))


@needs_encoder
def test_recorder_empty_states_raise(tmp_path):
    r = VideoRecorder(tmp_path, KEY, FPS)
    with pytest.raises(RuntimeError, match="no frames"):
        r.finalize_episode()
    with pytest.raises(RuntimeError):
        r.feature_info()
    with pytest.raises(RuntimeError):
        r.feature_stats()


@needs_encoder
def test_recorder_layout_and_chunk_rollover(tmp_path):
    """3 episodes at chunks_size=2: files land at chunk-000/file-000,
    chunk-000/file-001, chunk-001/file-000 (lerobot's
    update_chunk_file_indices rule), one episode per mp4, from_timestamp 0.0
    and to_timestamp n/fps per episode."""
    r = VideoRecorder(tmp_path, KEY, FPS, chunks_size=2)
    lengths = [4, 6, 8]
    for n in lengths:
        for f in _frames(n, seed=n):
            r.append(f)
        r.finalize_episode()
    assert r.total_episodes == 3
    expect = [(0, 0), (0, 1), (1, 0)]
    for meta, n, (chunk, file) in zip(r.episode_metadata, lengths, expect):
        assert meta[f"videos/{KEY}/chunk_index"] == chunk
        assert meta[f"videos/{KEY}/file_index"] == file
        assert meta[f"videos/{KEY}/from_timestamp"] == 0.0
        assert meta[f"videos/{KEY}/to_timestamp"] == n / FPS
        rel = DEFAULT_VIDEO_PATH.format(video_key=KEY, chunk_index=chunk, file_index=file)
        assert (tmp_path / rel).is_file()


@needs_encoder
def test_recorder_exact_pixel_stats(tmp_path):
    """Two constant-value episodes (51/255 = 0.2 and 204/255 = 0.8 exactly)
    -> min 0.2, max 0.8, mean 0.5, std 0.3 per channel, nested (c, 1, 1),
    count = total frames — exact population stats, no sampling."""
    r = VideoRecorder(tmp_path, KEY, FPS)
    for value, n in ((51, 3), (204, 3)):
        for _ in range(n):
            r.append(np.full((H, W, 3), value, dtype=np.uint8))
        r.finalize_episode()
    s = r.feature_stats()
    assert np.asarray(s["min"]).shape == (3, 1, 1)
    np.testing.assert_allclose(np.asarray(s["min"]).ravel(), 0.2, atol=1e-12)
    np.testing.assert_allclose(np.asarray(s["max"]).ravel(), 0.8, atol=1e-12)
    np.testing.assert_allclose(np.asarray(s["mean"]).ravel(), 0.5, atol=1e-12)
    np.testing.assert_allclose(np.asarray(s["std"]).ravel(), 0.3, atol=1e-12)
    assert s["count"] == [6]

    info = r.feature_info()
    assert info["dtype"] == "video" and info["shape"] == [H, W, 3]
    assert info["names"] == ["height", "width", "channels"]
    assert info["info"]["video.codec"] == "av1"  # default libsvtav1's canonical name
    assert info["info"]["video.pix_fmt"] == "yuv420p"


# ------------------------------------------------- attach_video_metadata


@pytest.fixture(scope="module")
def robot():
    return caliper.Robot.from_urdf(_resolve_urdf("planner", None))


def _vector_ds(root, robot, lengths, fps=FPS):
    """A finalized vector-only RecorderV3 dataset (state/action only — a
    dtype-"video" key must have NO data-parquet column) with the given
    per-episode frame counts. Deterministic joint values."""
    rec = caliper.RecorderV3(robot, str(root), fps=fps)
    nd = robot.ndof
    for ep, n in enumerate(lengths):
        rec.start_episode(f"ep {ep}")
        for k in range(n):
            q = [0.1 * np.sin(k / 5.0 + ep)] * nd
            rec.append(q, q, k / fps)
        rec.finalize_episode()
    return rec.close()


def _video_recorder(root, lengths, key=KEY, seed=20):
    r = VideoRecorder(root, key, FPS)
    for ep, n in enumerate(lengths):
        for f in _frames(n, seed=seed + ep):
            r.append(f)
        r.finalize_episode()
    return r


@needs_encoder
def test_attach_rejects_bad_inputs(tmp_path, robot):
    lengths = [4, 5]
    root = _vector_ds(tmp_path / "ds", robot, lengths)
    with pytest.raises(ValueError, match="no recorders"):
        attach_video_metadata(root, [])
    dup = _video_recorder(root, lengths)
    with pytest.raises(ValueError, match="duplicate"):
        attach_video_metadata(root, [dup, dup])
    short = _video_recorder(tmp_path / "elsewhere", [4])  # 1 episode vs 2
    with pytest.raises(ValueError, match="episodes"):
        attach_video_metadata(root, [short])
    taken = _video_recorder(root, lengths, key="observation.state", seed=40)
    with pytest.raises(ValueError, match="already declared"):
        attach_video_metadata(root, [taken])


@needs_encoder
def test_attach_requires_every_video_on_disk(tmp_path, robot):
    lengths = [4, 5]
    root = _vector_ds(tmp_path / "ds", robot, lengths)
    vrec = _video_recorder(root, lengths)
    victim = pathlib.Path(root) / DEFAULT_VIDEO_PATH.format(
        video_key=KEY, chunk_index=0, file_index=1
    )
    victim.rename(victim.with_suffix(".hidden"))
    with pytest.raises(FileNotFoundError):
        attach_video_metadata(root, [vrec])
    # crash-safe ordering: the failed attach must not have touched meta/
    victim.with_suffix(".hidden").rename(victim)
    attach_video_metadata(root, [vrec])  # now clean
    with pytest.raises(ValueError, match="already has video metadata"):
        attach_video_metadata(root, [vrec])  # double-attach guard


# --------------------------------- THE GATE: real lerobot decodes our videos


@pytest.fixture(scope="module")
def video_ds(tmp_path_factory, robot):
    """Vector RecorderV3 dataset + synthetic per-episode mp4s + attach —
    collected once, loaded by real lerobot in the gates below."""
    if not _OK:
        pytest.skip(f"no video encoder: {_REASON}")
    lengths = [12, 15]
    root = _vector_ds(tmp_path_factory.mktemp("vds") / "ds", robot, lengths)
    vrec = _video_recorder(root, lengths, seed=20)
    attach_video_metadata(root, [vrec])
    return root, lengths


@needs_encoder
def test_lerobot_loads_and_decodes_within_tolerance(video_ds):
    """A dataset assembled by this module is a REAL LeRobotDataset: dtype
    "video" feature, frames decoded from the mp4s come back CHW float32 in
    [0, 1] and match the originally stored pixels (mean abs diff < 0.06 —
    measured ~0.013, see module docstring)."""
    pytest.importorskip("lerobot", reason="lerobot not installed")
    torch = pytest.importorskip("torch")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, lengths = video_ds
    ds = LeRobotDataset("caliper/video_gate", root=str(root))
    assert ds.meta.total_episodes == len(lengths)
    feat = ds.meta.features[KEY]
    assert feat["dtype"] == "video"
    assert list(feat["shape"]) == [H, W, 3]
    assert ds.meta.info["video_path"] == DEFAULT_VIDEO_PATH
    # the four per-episode video columns are what _query_videos navigates by
    ep0 = ds.meta.episodes[0]
    assert ep0[f"videos/{KEY}/from_timestamp"] == 0.0
    assert ep0[f"videos/{KEY}/to_timestamp"] == lengths[0] / FPS

    idx = 0
    worst = 0.0
    for ep, n in enumerate(lengths):
        for k in range(n):
            item = ds[idx]
            idx += 1
            img = item[KEY]
            assert tuple(img.shape) == (3, H, W) and img.dtype == torch.float32
            assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
            orig = torch.from_numpy(_frames(n, seed=20 + ep)[k]).permute(2, 0, 1) / 255.0
            worst = max(worst, float((img - orig.float()).abs().mean()))
    assert idx == len(ds) == sum(lengths)
    assert worst < 0.06, f"decoded frames diverged from stored pixels: {worst:.4f}"


@needs_encoder
def test_sim_camera_video_dataset_matches_png_twin(tmp_path):
    """END-TO-END: `collect_camera_dataset(video=True)` (2 eps, 64x64) loads
    through real lerobot and its decoded frames match the dtype-"image" twin
    collected with the SAME seed (MuJoCo offscreen renders are byte-stable,
    so the PNG dataset IS the originally rendered pixels) within codec
    tolerance; the video dataset's data parquet carries NO camera column."""
    pytest.importorskip("mujoco", reason="mujoco not installed")
    pytest.importorskip("lerobot", reason="lerobot not installed")
    pq = pytest.importorskip("pyarrow.parquet")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from caliper_learn.collect_sim import collect_camera_dataset

    kwargs = dict(n_episodes=2, fps=FPS, seed0=7, width=W, height=H, max_frames=10)
    vroot = collect_camera_dataset(str(tmp_path / "video"), video=True, **kwargs)
    iroot = collect_camera_dataset(str(tmp_path / "image"), **kwargs)

    # one episode per mp4, exactly as promised
    mp4s = sorted(p.relative_to(vroot) for p in pathlib.Path(vroot).glob("videos/**/*.mp4"))
    assert [str(p) for p in mp4s] == [
        f"videos/{KEY}/chunk-000/file-000.mp4",
        f"videos/{KEY}/chunk-000/file-001.mp4",
    ]
    # a video key must NOT exist in the data parquet (lerobot skips it there)
    data_files = sorted(pathlib.Path(vroot).glob("data/*/*.parquet"))
    assert data_files, "no data parquet written"
    assert all(KEY not in pq.read_schema(f).names for f in data_files)

    dv = LeRobotDataset("caliper/sim_video", root=vroot)
    di = LeRobotDataset("caliper/sim_image", root=iroot)
    assert dv.meta.features[KEY]["dtype"] == "video"
    assert di.meta.features[KEY]["dtype"] == "image"
    assert len(dv) == len(di) > 0
    worst = max(
        float((dv[i][KEY] - di[i][KEY]).abs().mean()) for i in range(len(dv))
    )
    assert worst < 0.05, f"video frames diverged from rendered pixels: {worst:.4f}"
