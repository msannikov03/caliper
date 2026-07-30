"""dtype-"video" storage tests: the encode chain, the per-episode mp4 layout,
the NATIVE writer path (video features declared on `caliper.RecorderV3`), the
`attach_video_metadata` repair tool, the equality of the two (the retirement
proof), and THE GATE — a dataset this module wrote loads through REAL lerobot
(`LeRobotDataset`) and hands back decoded video frames (CHW float32) matching
the originally stored pixels within codec tolerance.

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


# ------------------------------------------------- native-writer accessors


@needs_encoder
def test_feature_spec_needs_a_known_shape(tmp_path):
    """`feature_spec()` describes the dataset feature BEFORE any frame is
    recorded, so the shape has to come from the constructor — asking without
    one is an error, not a guess."""
    r = VideoRecorder(tmp_path, KEY, FPS)
    with pytest.raises(RuntimeError, match="frame shape unknown"):
        r.feature_spec()
    r.append(_frames(1)[0])
    assert r.feature_spec() == (KEY, H, W, 3, "av1", "yuv420p")

    declared = VideoRecorder(tmp_path / "d", KEY, FPS, height=H, width=W, codec="h264")
    assert declared.feature_spec() == (KEY, H, W, 3, "h264", "yuv420p")
    # a declared shape also locks what append() accepts
    with pytest.raises(ValueError, match="locked shape"):
        declared.append(np.zeros((H, W * 2, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "kwargs",
    [{"height": H}, {"width": W}, {"height": 0, "width": W}, {"height": H - 1, "width": W}],
    ids=["h-only", "w-only", "zero", "odd"],
)
def test_declared_shape_is_validated(tmp_path, kwargs):
    with pytest.raises(ValueError):
        VideoRecorder(tmp_path, KEY, FPS, **kwargs)


@needs_encoder
def test_last_slot_and_flat_stats(tmp_path):
    r = VideoRecorder(tmp_path, KEY, FPS, chunks_size=2)
    with pytest.raises(RuntimeError, match="no episodes"):
        r.last_slot()
    for value, n in ((51, 4), (204, 6)):
        for _ in range(n):
            r.append(np.full((H, W, 3), value, dtype=np.uint8))
        r.finalize_episode()
        # the slot names the register_episode_video() keyword arguments
        assert set(r.last_slot()) == {
            "chunk_index", "file_index", "from_timestamp", "to_timestamp"
        }
    assert r.last_slot() == {
        "chunk_index": 0, "file_index": 1, "from_timestamp": 0.0, "to_timestamp": 6 / FPS,
    }
    flat = r.feature_stats_flat()
    nested = r.feature_stats()
    assert flat["count"] == nested["count"] == [10]
    for k in ("min", "max", "mean", "std"):
        assert np.asarray(flat[k]).shape == (3,)
        np.testing.assert_array_equal(flat[k], np.asarray(nested[k]).ravel())


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


def _native_ds(root, robot, lengths, key=KEY, seed=20, fps=FPS):
    """The NATIVE path: the camera is a `video_features` entry on
    `RecorderV3`, each episode's mp4 slot is registered on the recorder, and
    the pixel stats go in before close — no post-write rewrite anywhere.
    Records the same joint values and the same frames `_vector_ds` /
    `_video_recorder` do, so the two paths are comparable."""
    vrec = VideoRecorder(root, key, fps, height=H, width=W)
    rec = caliper.RecorderV3(
        robot, str(root), fps=fps, video_features=[vrec.feature_spec()]
    )
    nd = robot.ndof
    for ep, n in enumerate(lengths):
        rec.start_episode(f"ep {ep}")
        for k, frame in enumerate(_frames(n, seed=seed + ep)):
            q = [0.1 * np.sin(k / 5.0 + ep)] * nd
            rec.append(q, q, k / fps)
            vrec.append(frame)
        vrec.finalize_episode()
        rec.register_episode_video(key, **vrec.last_slot())
        rec.finalize_episode()
    rec.set_video_stats(key, **vrec.feature_stats_flat())
    return rec.close(), vrec


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


@needs_encoder
def test_attach_rejects_fps_mismatch(tmp_path, robot):
    """A recorder encoded at a different fps than the dataset must be refused
    — the mp4 clock and the parquet timestamps would silently desync (1.5x
    A/V drift with nobody the wiser)."""
    lengths = [4, 5]
    root = _vector_ds(tmp_path / "ds", robot, lengths)
    r = VideoRecorder(root, KEY, FPS * 2)
    for ep, n in enumerate(lengths):
        for f in _frames(n, seed=30 + ep):
            r.append(f)
        r.finalize_episode()
    with pytest.raises(ValueError, match="fps"):
        attach_video_metadata(root, [r])


@needs_encoder
def test_attach_rejects_frame_count_mismatch(tmp_path, robot):
    """Same episode COUNT but a wrong per-episode frame count must be refused
    — the video would run short/long against the frame data. Nothing in meta/
    may be touched by the failed attach."""
    import json

    import pyarrow.parquet as pq

    root = _vector_ds(tmp_path / "ds", robot, [4, 5])
    r = _video_recorder(root, [4, 4])  # episode 1: 4 video frames vs 5 rows
    with pytest.raises(ValueError, match="video"):
        attach_video_metadata(root, [r])
    ep_file = next(iter(pathlib.Path(root).glob("meta/episodes/*/*.parquet")))
    assert not any(c.startswith("videos/") for c in pq.read_schema(ep_file).names)
    info = json.loads((pathlib.Path(root) / "meta" / "info.json").read_text())
    assert not info.get("video_path")


@needs_encoder
def test_attach_crash_leaves_meta_intact(tmp_path, robot, monkeypatch):
    """Crash-safety regression: a failure mid-parquet-write used to truncate
    the ONLY episodes parquet in place. With temp + os.replace the original
    must stay byte-identical, no temp litter, and a retry must succeed."""
    import pyarrow.parquet as pq

    lengths = [4, 5]
    root = _vector_ds(tmp_path / "ds", robot, lengths)
    vrec = _video_recorder(root, lengths)
    ep_file = next(iter(pathlib.Path(root).glob("meta/episodes/*/*.parquet")))
    before = ep_file.read_bytes()

    real_write = pq.write_table

    def dying_write(table, where, **kw):
        # Simulate a crash mid-write: partial bytes land at the destination
        # path, then the process "dies".
        pathlib.Path(where).write_bytes(b"partial garbage")
        raise RuntimeError("simulated crash during parquet write")

    monkeypatch.setattr(pq, "write_table", dying_write)
    with pytest.raises(RuntimeError, match="simulated crash"):
        attach_video_metadata(root, [vrec])
    monkeypatch.setattr(pq, "write_table", real_write)

    assert ep_file.read_bytes() == before, "episodes parquet was corrupted in place"
    assert not list(pathlib.Path(root).glob("meta/**/*.tmp-attach"))
    attach_video_metadata(root, [vrec])  # original intact -> retry succeeds
    assert any(
        c.startswith("videos/") for c in pq.read_schema(ep_file).names
    )


# ------------------------------------------ native path vs the repair tool


@needs_encoder
def test_native_and_bridge_write_the_same_metadata(tmp_path, robot):
    """THE RETIREMENT PROOF: the same recording written natively (video
    feature on `RecorderV3`) and via the post-write bridge produces byte-equal
    `meta/episodes` (schema AND every row), the same `info.json` feature entry
    and `video_path`, and the same `stats.json` entry. Only the mp4 BYTES may
    differ (multi-threaded encoders), never the metadata."""
    import json

    import pyarrow.parquet as pq

    lengths = [12, 15]
    native_root, _ = _native_ds(tmp_path / "native", robot, lengths)
    bridge_root = _vector_ds(tmp_path / "bridge", robot, lengths)
    attach_video_metadata(bridge_root, [_video_recorder(bridge_root, lengths)])

    def episodes(root):
        return pq.read_table(next(iter(pathlib.Path(root).glob("meta/episodes/*/*.parquet"))))

    nat, bri = episodes(native_root), episodes(bridge_root)
    assert nat.schema.names == bri.schema.names
    assert nat.schema.types == bri.schema.types
    assert nat.equals(bri), "native and bridged meta/episodes must be identical"

    def meta(root, name):
        return json.loads((pathlib.Path(root) / "meta" / name).read_text())

    ninfo, binfo = meta(native_root, "info.json"), meta(bridge_root, "info.json")
    assert ninfo["features"][KEY] == binfo["features"][KEY]
    assert ninfo["video_path"] == binfo["video_path"] == DEFAULT_VIDEO_PATH
    assert ninfo["features"].keys() == binfo["features"].keys()
    assert meta(native_root, "stats.json")[KEY] == meta(bridge_root, "stats.json")[KEY]


def _open_native(root, robot, n_frames=6):
    """A native recorder + video recorder mid-episode, `n_frames` frames in
    on both sides (nothing registered yet)."""
    vrec = VideoRecorder(root, KEY, FPS, height=H, width=W)
    rec = caliper.RecorderV3(robot, str(root), fps=FPS, video_features=[vrec.feature_spec()])
    nd = robot.ndof
    rec.start_episode("ep 0")
    for k, frame in enumerate(_frames(n_frames)):
        rec.append([0.0] * nd, [0.0] * nd, k / FPS)
        vrec.append(frame)
    return rec, vrec


@needs_encoder
def test_native_writer_requires_registration_before_finalize(tmp_path, robot):
    """An episode may not close without its video registered — and the refusal
    keeps the episode OPEN (frames intact), so registering and retrying works.
    Closing without pixel stats is refused too."""
    rec, vrec = _open_native(tmp_path / "ds", robot)
    with pytest.raises(ValueError, match="no registration"):
        rec.finalize_episode()
    # the camera key is not a frame payload — its frames are in the mp4
    with pytest.raises(ValueError, match="video feature"):
        rec.append([0.0] * robot.ndof, [0.0] * robot.ndof, 6 / FPS, images={KEY: b"png"})

    vrec.finalize_episode()
    rec.register_episode_video(KEY, **vrec.last_slot())
    rec.finalize_episode()  # the retry lands: the frames were never lost
    with pytest.raises(ValueError, match="no pixel stats"):
        rec.close()


@needs_encoder
def test_native_writer_refuses_a_desynced_span(tmp_path, robot):
    """A span covering fewer frames than the episode holds is refused at save
    — the desync the bridge caught by comparing frame counts."""
    rec, vrec = _open_native(tmp_path / "ds", robot)
    vrec.finalize_episode()
    rec.register_episode_video(KEY, **{**vrec.last_slot(), "to_timestamp": 3 / FPS})
    with pytest.raises(ValueError, match="desync"):
        rec.finalize_episode()


@needs_encoder
def test_native_writer_requires_the_mp4_on_disk(tmp_path, robot):
    """A registered episode whose mp4 is missing is refused at close — the
    check the bridge did before it would touch `meta/`."""
    rec, vrec = _open_native(tmp_path / "ds", robot)
    vrec.finalize_episode()
    rec.register_episode_video(KEY, **vrec.last_slot())
    rec.finalize_episode()
    rec.set_video_stats(KEY, **vrec.feature_stats_flat())

    victim = pathlib.Path(tmp_path / "ds") / DEFAULT_VIDEO_PATH.format(
        video_key=KEY, chunk_index=0, file_index=0
    )
    victim.rename(victim.with_suffix(".hidden"))
    with pytest.raises(ValueError, match="does not exist"):
        rec.close()


# --------------------------------- THE GATE: real lerobot decodes our videos


@pytest.fixture(scope="module", params=["native", "bridge"])
def video_ds(request, tmp_path_factory, robot):
    """A dtype-"video" dataset holding the same episodes, written both ways —
    natively (the default path) and through the repair tool — so the lerobot
    gate below runs over BOTH."""
    if not _OK:
        pytest.skip(f"no video encoder: {_REASON}")
    lengths = [12, 15]
    root = tmp_path_factory.mktemp(f"vds_{request.param}") / "ds"
    if request.param == "native":
        root, _ = _native_ds(root, robot, lengths, seed=20)
    else:
        root = _vector_ds(root, robot, lengths)
        attach_video_metadata(root, [_video_recorder(root, lengths, seed=20)])
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
