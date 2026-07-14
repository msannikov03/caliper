"""MP4 video encoding for LeRobotDataset v3.0 — the `dtype: "video"` half of
camera storage that the Rust writer does not cover yet.

`caliper.RecorderV3` stores cameras as `dtype: "image"` (PNG bytes embedded in
the data parquet). Real lerobot datasets more commonly store cameras as
`dtype: "video"`: the frames live in `videos/{key}/chunk-XXX/file-XXX.mp4`,
the data parquet carries NO column for the key at all (lerobot's
`get_hf_features_from_features` skips dtype "video"), and `meta/episodes`
grows four columns per key — `videos/{key}/chunk_index`, `.../file_index`,
`.../from_timestamp`, `.../to_timestamp` — which `LeRobotDataset._query_videos`
uses to locate the file and decode at `from_timestamp + frame_timestamp`.

This module is the python-side bridge until the Rust writer grows video
columns (NO Rust changes in this wave — documented deferral):

    caliper.RecorderV3 (vector features only)   # data + meta skeleton
    + VideoRecorder (camera frames -> mp4)      # videos/{key}/... files
    -> attach_video_metadata(root, [vrec, ..])  # post-write pyarrow rewrite

ENCODE SETTINGS mirror lerobot 0.4.4 `encode_video_frames`
(`lerobot/datasets/video_utils.py`): default vcodec `libsvtav1`,
`pix_fmt="yuv420p"`, GOP `g=2`, `crf=30`, and svtav1 `preset=12` — byte-for-
byte the `_get_codec_options` output for the two software codecs supported
here (`libsvtav1`, `h264`; lerobot's HW-encoder auto-probe is out of scope).

ENCODER CHAIN (probed by `available()`, never assumed): PyAV first (it is how
lerobot itself encodes), else a `ffmpeg` subprocess fed rawvideo over stdin,
else honestly unavailable — callers get `(False, reason)` / a loud error, not
a silently image-only dataset.

LAYOUT SHIPPED: ONE EPISODE PER MP4 FILE, `file_index` advancing per episode
by lerobot's own `update_chunk_file_indices` rule, `from_timestamp = 0.0` and
`to_timestamp = n_frames / fps` for every episode. This is a valid instance
of the format — the reader trusts the per-episode columns and never assumes
multi-episode files, and lerobot's own writer produces exactly this shape
whenever an episode crosses `video_files_size_in_mb` (the `shutil.move`
branch of `_save_episode_video`). Concatenating episodes into shared files up
to the 200 MB target (lerobot's default-path space optimization) is
deliberately NOT reimplemented here.

Determinism: input frames are the caller's (seeded) pixels and the layout is
a pure function of them, but the encoded BYTES are not guaranteed bit-stable
across runs (SVT-AV1/x264 encode multi-threaded); tests must compare decoded
content, not file hashes.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 — fixed argv, no shell
from pathlib import Path

import numpy as np

DEFAULT_CODEC = "libsvtav1"
#: v3.0 video path template (`lerobot.datasets.utils.DEFAULT_VIDEO_PATH`),
#: written into `info.json` by `attach_video_metadata`.
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

# lerobot's encode defaults (video_utils.py: encode_video_frames +
# _get_codec_options for software codecs).
_PIX_FMT = "yuv420p"
_GOP = 2
_CRF = 30
_SVTAV1_PRESET = 12
# lerobot.datasets.utils.DEFAULT_CHUNK_SIZE — max files per chunk-XXX dir.
_CHUNKS_SIZE = 1000

# `info.json` `features[key]["info"]["video.codec"]` values: the container-
# level canonical codec names lerobot's `get_video_info` reads back via
# `stream.codec.canonical_name` (verified on files encoded by this module:
# libsvtav1 -> "av1", h264 -> "h264").
_CANONICAL = {"libsvtav1": "av1", "h264": "h264"}
# ffmpeg-CLI encoder names for the subprocess fallback. PyAV resolves the
# plain "h264" encoder alias to libx264 itself; the CLI needs it explicit.
_FFMPEG_ENCODER = {"libsvtav1": "libsvtav1", "h264": "libx264"}


def available(codec: str = DEFAULT_CODEC) -> tuple[bool, str]:
    """Probe the encoder chain for `codec` -> `(ok, reason)`.

    Chain: PyAV with the named encoder compiled in -> `ffmpeg` subprocess
    whose `-encoders` list has it -> `(False, why)`. `reason` names the
    backend that will encode (`"pyav (...)"` / `"ffmpeg subprocess (...)"`)
    or spells out exactly what is missing — callers skip or fail LOUDLY on
    it, they never fake availability.
    """
    if codec not in _CANONICAL:
        return False, f"unsupported codec '{codec}' (supported: {sorted(_CANONICAL)})"
    try:
        import av  # lazy: keep `import caliper_learn` av-free

        try:
            av.codec.Codec(codec, "w")
            return True, f"pyav ({codec})"
        except Exception as e:  # av.codec.codec.UnknownCodecError, keep broad
            pyav_reason = f"pyav importable but lacks encoder '{codec}' ({e})"
    except ImportError as e:
        pyav_reason = f"pyav not importable ({e})"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False, f"{pyav_reason}; no ffmpeg on PATH"
    enc = _FFMPEG_ENCODER[codec]
    try:
        out = subprocess.run(  # nosec B603 — fixed argv
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except Exception as e:
        return False, f"{pyav_reason}; ffmpeg probe failed ({e})"
    if f" {enc} " in out:
        return True, f"ffmpeg subprocess ({enc}); {pyav_reason}"
    return False, f"{pyav_reason}; ffmpeg at {ffmpeg} lacks encoder '{enc}'"


def _validate_frames(frames) -> np.ndarray:
    """Frames -> a validated `(n, h, w, 3)` uint8 array (rejects ragged input,
    wrong dtype/channels, and odd dimensions — yuv420p halves chroma, so odd
    h/w cannot encode)."""
    try:
        arr = np.asarray(frames)
    except ValueError as e:  # ragged list of differently-shaped frames
        raise ValueError(f"frames must all share one (h, w, 3) shape: {e}") from e
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"expected (n, h, w, 3) HWC RGB frames, got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("no frames to encode")
    if arr.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {arr.dtype}")
    if arr.shape[1] % 2 or arr.shape[2] % 2:
        raise ValueError(
            f"frame size {arr.shape[1]}x{arr.shape[2]} has an odd dimension; "
            f"pix_fmt {_PIX_FMT} needs even height and width"
        )
    return np.ascontiguousarray(arr)


def encode_episode_video(
    frames_hwc_u8,
    fps: int,
    out_path: str | Path,
    codec: str = DEFAULT_CODEC,
) -> float:
    """Encode one episode's frames (`(n, h, w, 3)` uint8 RGB, or a list of
    `(h, w, 3)` frames) into `out_path` with lerobot's settings (see module
    docstring), via the `available()` chain. Returns the episode duration
    `n / fps` in seconds (exact by construction — every frame is stamped at
    `k / fps`, matching the parquet timestamps `RecorderV3` stores).

    Refuses to overwrite an existing file (lerobot's encode silently SKIPS on
    collision — here that would drop an episode's video, so it is an error).
    Note libsvtav1's own floor is 64x64; smaller frames fail in the encoder.
    """
    arr = _validate_frames(frames_hwc_u8)
    fps = int(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    ok, reason = available(codec)
    if not ok:
        raise RuntimeError(f"no video encoder available: {reason}")
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite existing video: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if reason.startswith("pyav"):
        _encode_pyav(arr, fps, out_path, codec)
    else:
        _encode_ffmpeg(arr, fps, out_path, codec)
    if not out_path.exists():  # mirrors lerobot's post-encode existence check
        raise OSError(f"video encoding produced no file: {out_path}")
    return arr.shape[0] / fps


def _codec_options(codec: str) -> dict[str, str]:
    """lerobot `_get_codec_options(vcodec, g=2, crf=30, preset=None)` for the
    software codecs: `{"g": "2", "crf": "30"}` (+ `preset: "12"` for svtav1,
    its `preset if preset is not None else "12"` default)."""
    options = {"g": str(_GOP), "crf": str(_CRF)}
    if codec == "libsvtav1":
        options["preset"] = str(_SVTAV1_PRESET)
    return options


def _encode_pyav(arr: np.ndarray, fps: int, out_path: Path, codec: str) -> None:
    """The lerobot path: `av.open(w) -> add_stream(codec, fps, options)`,
    frame-by-frame encode + mux, then a flush — same calls its
    `encode_video_frames` makes (from ndarray instead of PNGs on disk)."""
    import av

    with av.open(str(out_path), "w") as out:
        stream = out.add_stream(codec, fps, options=_codec_options(codec))
        stream.pix_fmt = _PIX_FMT
        stream.width = arr.shape[2]
        stream.height = arr.shape[1]
        for img in arr:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():  # flush the encoder
            out.mux(packet)


def _encode_ffmpeg(arr: np.ndarray, fps: int, out_path: Path, codec: str) -> None:
    """Fallback: pipe rawvideo rgb24 into a `ffmpeg` subprocess with the same
    codec options. Only reached when PyAV cannot encode `codec`."""
    n, h, w, _ = arr.shape
    opts = _codec_options(codec)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", _FFMPEG_ENCODER[codec], "-pix_fmt", _PIX_FMT,
        "-g", opts["g"], "-crf", opts["crf"],
    ]
    if "preset" in opts:
        cmd += ["-preset", opts["preset"]]
    cmd.append(str(out_path))
    proc = subprocess.run(cmd, input=arr.tobytes(), capture_output=True)  # nosec B603
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"ffmpeg encode failed (exit {proc.returncode}): {tail}")


class VideoRecorder:
    """Per-episode camera buffer -> the v3.0 `videos/{key}/...` mp4 layout;
    the camera-side companion of a VECTOR-ONLY `caliper.RecorderV3` (declare
    no `image_features` — a dtype-"video" key must have NO data-parquet
    column, or lerobot's schema-checked load rejects the dataset).

    Cadence contract: `append(frame)` once per `RecorderV3.append(...)` call,
    `finalize_episode()` right after `RecorderV3.finalize_episode()`, and
    after `RecorderV3.close()` hand every recorder to
    `attach_video_metadata(root, recorders)` — nothing is registered in
    `meta/` until that post-write step. Also folds per-channel pixel stats
    (lerobot's [0, 1] scale) across all frames for `meta/stats.json`.
    """

    def __init__(
        self,
        root: str | Path,
        video_key: str,
        fps: int,
        *,
        codec: str = DEFAULT_CODEC,
        chunks_size: int = _CHUNKS_SIZE,
    ):
        ok, reason = available(codec)
        if not ok:
            raise RuntimeError(f"no video encoder available: {reason}")
        fps = int(fps)
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if int(chunks_size) <= 0:
            raise ValueError(f"chunks_size must be positive, got {chunks_size}")
        if not video_key or "/" in video_key or "\\" in video_key:
            raise ValueError(
                f"video_key must be a non-empty name without path separators, "
                f"got {video_key!r}"
            )
        self._root = Path(root)
        self._key = str(video_key)
        self._fps = fps
        self._codec = codec
        self._chunks_size = int(chunks_size)
        self._frames: list[np.ndarray] = []
        self._episodes: list[dict] = []
        self._chunk = 0
        self._file = 0
        self._shape: tuple[int, int] | None = None  # (h, w), locked at frame 1
        # Per-channel running pixel stats in [0, 1] (mirrors the Rust writer's
        # fold_image_stats): exact population stats over EVERY pixel.
        self._stat_min = np.full(3, np.inf)
        self._stat_max = np.full(3, -np.inf)
        self._stat_sum = np.zeros(3)
        self._stat_sumsq = np.zeros(3)
        self._stat_pixels = 0
        self._stat_frames = 0

    @property
    def video_key(self) -> str:
        return self._key

    @property
    def total_episodes(self) -> int:
        return len(self._episodes)

    @property
    def episode_metadata(self) -> list[dict]:
        """One dict per finalized episode: the four `videos/{key}/...` columns
        `attach_video_metadata` appends to `meta/episodes`."""
        return [dict(m) for m in self._episodes]

    def append(self, frame_hwc_u8) -> None:
        """Buffer one `(h, w, 3)` uint8 RGB frame (copied). The first frame
        locks the shape; later mismatches raise before anything is encoded."""
        f = np.asarray(frame_hwc_u8)
        if f.ndim != 3 or f.shape[-1] != 3:
            raise ValueError(f"expected one (h, w, 3) RGB frame, got shape {f.shape}")
        if f.dtype != np.uint8:
            raise ValueError(f"frame must be uint8, got {f.dtype}")
        if f.shape[0] % 2 or f.shape[1] % 2:
            raise ValueError(
                f"frame size {f.shape[0]}x{f.shape[1]} has an odd dimension; "
                f"pix_fmt {_PIX_FMT} needs even height and width"
            )
        hw = (int(f.shape[0]), int(f.shape[1]))
        if self._shape is None:
            self._shape = hw
        elif hw != self._shape:
            raise ValueError(f"frame shape {hw} != locked shape {self._shape}")
        self._frames.append(f.copy())

    def finalize_episode(self) -> str:
        """Encode the buffered frames as this episode's mp4 (one episode per
        file — see module docstring), record its metadata row, advance the
        (chunk, file) indices by lerobot's `update_chunk_file_indices` rule,
        and clear the buffer. Returns the relative video path written."""
        if not self._frames:
            raise RuntimeError("no frames buffered; call append(frame) first")
        rel = DEFAULT_VIDEO_PATH.format(
            video_key=self._key, chunk_index=self._chunk, file_index=self._file
        )
        arr = np.stack(self._frames)
        duration = encode_episode_video(
            arr, self._fps, self._root / rel, codec=self._codec
        )
        self._episodes.append({
            f"videos/{self._key}/chunk_index": self._chunk,
            f"videos/{self._key}/file_index": self._file,
            f"videos/{self._key}/from_timestamp": 0.0,
            f"videos/{self._key}/to_timestamp": duration,
        })
        x = arr.astype(np.float64) / 255.0
        self._stat_min = np.minimum(self._stat_min, x.min(axis=(0, 1, 2)))
        self._stat_max = np.maximum(self._stat_max, x.max(axis=(0, 1, 2)))
        self._stat_sum += x.sum(axis=(0, 1, 2))
        self._stat_sumsq += (x * x).sum(axis=(0, 1, 2))
        self._stat_pixels += arr.shape[0] * arr.shape[1] * arr.shape[2]
        self._stat_frames += arr.shape[0]
        # lerobot update_chunk_file_indices: wrap into the next chunk dir
        # after chunks_size files.
        if self._file == self._chunks_size - 1:
            self._chunk += 1
            self._file = 0
        else:
            self._file += 1
        self._frames.clear()
        return rel

    def feature_info(self) -> dict:
        """The `info.json` `features[key]` entry: dtype "video", HWC shape,
        the `["height", "width", "channels"]` names lerobot's
        `dataset_to_policy_features` requires, and the `info` sub-dict its
        `get_video_info`/`update_video_info` would read from the file
        (video.codec = the container canonical name, see `_CANONICAL`)."""
        if self._shape is None or not self._episodes:
            raise RuntimeError("no episodes finalized; nothing to describe")
        h, w = self._shape
        return {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "fps": self._fps,
            "info": {
                "video.height": h,
                "video.width": w,
                "video.codec": _CANONICAL[self._codec],
                "video.pix_fmt": _PIX_FMT,
                "video.is_depth_map": False,
                "video.fps": self._fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    def feature_stats(self) -> dict:
        """The `meta/stats.json` entry: per-channel min/max/mean/std nested
        `(c, 1, 1)` (broadcastable against CHW tensors, exactly the shape the
        Rust writer emits for dtype-"image" features) + lerobot's
        `count = [n_frames]` convention. Exact over every stored pixel — no
        sampling."""
        if self._stat_pixels == 0:
            raise RuntimeError("no episodes finalized; no stats to report")
        mean = self._stat_sum / self._stat_pixels
        var = np.maximum(self._stat_sumsq / self._stat_pixels - mean * mean, 0.0)

        def nest(v: np.ndarray) -> list:
            return [[[float(x)]] for x in v]

        return {
            "min": nest(self._stat_min),
            "max": nest(self._stat_max),
            "mean": nest(mean),
            "std": nest(np.sqrt(var)),
            "count": [self._stat_frames],
        }


def attach_video_metadata(root: str | Path, recorders) -> None:
    """Post-write bridge: register `recorders`' videos in a finalized
    `RecorderV3` dataset's `meta/` — THE step that turns "mp4 files on disk"
    into a loadable dtype-"video" dataset. Explicitly the python-side bridge
    until the Rust writer grows video columns.

    Rewrites, in crash-safe order (info.json LAST — its `video_path` doubles
    as the commit marker and the double-attach guard):
      1. `meta/episodes/chunk-000/file-000.parquet`: appends the four
         `videos/{key}/...` columns per recorder (pyarrow, snappy — same
         compression the writer used). The Rust writer emits exactly one
         episodes parquet; more than one means an unknown writer, so bail.
      2. `meta/stats.json`: adds each key's `(c, 1, 1)` pixel stats.
      3. `meta/info.json`: adds each `features[key]` entry (dtype "video")
         and sets `video_path` to lerobot's `DEFAULT_VIDEO_PATH` template.

    Validates before touching anything: episode counts must match
    `info.total_episodes`, every referenced mp4 must exist, keys must be new
    (a key already in `features` — e.g. recorded as dtype "image" — is an
    error, not an upgrade), and a dataset with `video_path` already set is
    refused.
    """
    import pyarrow as pa  # lazy: lerobot-adjacent dep, not a core sidecar dep
    import pyarrow.parquet as pq

    root = Path(root)
    recorders = list(recorders)
    if not recorders:
        raise ValueError("no recorders to attach")
    keys = [r.video_key for r in recorders]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate video keys: {keys}")

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if info.get("video_path"):
        raise ValueError(f"dataset already has video metadata attached: {root}")
    n_eps = int(info["total_episodes"])
    for r in recorders:
        if r.total_episodes != n_eps:
            raise ValueError(
                f"recorder '{r.video_key}' has {r.total_episodes} episodes, "
                f"dataset has {n_eps} — every episode needs its video"
            )
        if r.video_key in info["features"]:
            raise ValueError(
                f"feature '{r.video_key}' already declared "
                f"(dtype {info['features'][r.video_key].get('dtype')!r}); "
                f"a video key must not exist in the data parquet"
            )
        for m in r.episode_metadata:
            rel = DEFAULT_VIDEO_PATH.format(
                video_key=r.video_key,
                chunk_index=m[f"videos/{r.video_key}/chunk_index"],
                file_index=m[f"videos/{r.video_key}/file_index"],
            )
            if not (root / rel).is_file():
                raise FileNotFoundError(f"episode video missing: {root / rel}")

    ep_files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if len(ep_files) != 1:
        raise ValueError(
            f"expected exactly one meta/episodes parquet (the Rust writer's "
            f"output), found {len(ep_files)} under {root / 'meta' / 'episodes'}"
        )
    table = pq.read_table(ep_files[0])
    if table.num_rows != n_eps:
        raise ValueError(
            f"meta/episodes has {table.num_rows} rows, info.json says {n_eps}"
        )
    for name in table.column_names:
        if name.startswith("videos/"):
            raise ValueError(f"meta/episodes already has video columns ({name})")
    for r in recorders:
        meta = r.episode_metadata
        for suffix, pa_type in (
            ("chunk_index", pa.int64()),
            ("file_index", pa.int64()),
            ("from_timestamp", pa.float64()),
            ("to_timestamp", pa.float64()),
        ):
            col = f"videos/{r.video_key}/{suffix}"
            table = table.append_column(
                pa.field(col, pa_type), pa.array([m[col] for m in meta], pa_type)
            )
    pq.write_table(table, ep_files[0], compression="snappy")

    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for r in recorders:
        if r.video_key in stats:
            raise ValueError(f"stats.json already has an entry for '{r.video_key}'")
        stats[r.video_key] = r.feature_stats()
    stats_path.write_text(json.dumps(dict(sorted(stats.items())), indent=2))

    for r in recorders:
        info["features"][r.video_key] = r.feature_info()
    info["features"] = dict(sorted(info["features"].items()))  # keep BTreeMap order
    info["video_path"] = DEFAULT_VIDEO_PATH
    info_path.write_text(json.dumps(info, indent=2))
