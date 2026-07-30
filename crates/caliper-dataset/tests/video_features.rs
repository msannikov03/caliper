//! Native `dtype: "video"` features through the public writer API: the
//! `info.json` entry (shape/names/`info` sub-dict/`video_path`), the four
//! `videos/{key}/*` columns of `meta/episodes`, the `(c, 1, 1)` pixel stats,
//! the absence of any data-parquet column, and every loud refusal that keeps
//! a video dataset from being written half-registered.
//!
//! The expectations here are pinned against what `caliper_learn.video`'s
//! pyarrow bridge wrote before the writer grew these columns (that bridge is
//! lerobot-parity-proven — `learn/tests/test_video.py` loads its output
//! through a real `LeRobotDataset`), so this file is the Rust half of the
//! retirement proof. No mp4 is ever encoded: the writer only ever checks that
//! the files the caller claims exist, so the fixtures below are plain bytes
//! at the right paths.

use arrow::array::{Array, Float64Array, Int64Array};
use arrow::datatypes::DataType;
use caliper_dataset::{
    DEFAULT_VIDEO_PATH, DatasetReader, DatasetSpec, DatasetWriter, FeatureSpec, FeatureStats,
    format_video_path,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs;
use std::path::{Path, PathBuf};

const KEY: &str = "observation.images.cam";
const H: usize = 64;
const W: usize = 48;
const FPS: u32 = 30;

fn tmpdir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("caliper_dataset_vid_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    p
}

fn spec() -> DatasetSpec {
    DatasetSpec::new(
        FPS,
        "cam_bot",
        vec![
            FeatureSpec::vector("observation.state", 2, Some(vec!["j1".into(), "j2".into()])),
            FeatureSpec::video(KEY, H, W, 3, "av1", "yuv420p"),
        ],
    )
}

/// Stand in for an encoded episode video: the writer never decodes, it only
/// requires the file the caller registered to be on disk at finalize.
fn touch_video(root: &Path, chunk: u64, file: u64) {
    let rel = format_video_path(DEFAULT_VIDEO_PATH, KEY, chunk, file).unwrap();
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, b"not really an mp4").unwrap();
}

fn add_frames(w: &mut DatasetWriter, n: usize) {
    for i in 0..n {
        let s = [0.1 * i as f64, 0.2 * i as f64];
        w.add_frame(&[("observation.state", &s)]).unwrap();
    }
}

fn pixel_stats() -> FeatureStats {
    FeatureStats {
        min: vec![0.0, 0.1, 0.2],
        max: vec![1.0, 0.9, 0.8],
        mean: vec![0.5, 0.4, 0.3],
        std: vec![0.25, 0.2, 0.15],
        count: vec![7],
    }
}

/// One episode per mp4, `file_index` advancing per episode — the layout
/// `caliper_learn.video.VideoRecorder` produces.
fn record_episodes(dir: &Path, lengths: &[usize]) -> PathBuf {
    let mut w = DatasetWriter::create(dir, spec()).unwrap();
    for (ep, &n) in lengths.iter().enumerate() {
        add_frames(&mut w, n);
        touch_video(dir, 0, ep as u64);
        w.register_episode_video(KEY, 0, ep as u64, 0.0, n as f64 / f64::from(FPS))
            .unwrap();
        w.save_episode(&format!("ep {ep}")).unwrap();
    }
    w.set_video_stats(KEY, pixel_stats()).unwrap();
    w.finalize().unwrap()
}

fn episodes_table(root: &Path) -> arrow::record_batch::RecordBatch {
    let path = root.join("meta/episodes/chunk-000/file-000.parquet");
    let file = fs::File::open(path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    reader.next().unwrap().unwrap()
}

#[test]
fn info_json_entry_matches_the_bridge() {
    let dir = tmpdir("info");
    let root = record_episodes(&dir, &[4, 3]);

    let info: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("meta/info.json")).unwrap()).unwrap();
    assert_eq!(info["video_path"], DEFAULT_VIDEO_PATH);
    // Byte-for-byte the entry `attach_video_metadata` used to splice in.
    assert_eq!(
        info["features"][KEY],
        serde_json::json!({
            "dtype": "video",
            "shape": [H, W, 3],
            "names": ["height", "width", "channels"],
            "fps": FPS,
            "info": {
                "video.height": H,
                "video.width": W,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": false,
            },
        })
    );
    // Vector features keep their plain entry — no stray `info` key.
    assert!(info["features"]["observation.state"].get("info").is_none());
    // `video_files_size_in_mb` is written for every dataset already; the
    // video-less path must not have grown a `video_path`.
    let plain = tmpdir("info_plain");
    let mut w = DatasetWriter::create(
        &plain,
        DatasetSpec::new(FPS, "bot", vec![FeatureSpec::vector("action", 2, None)]),
    )
    .unwrap();
    add_frames_named(&mut w, "action", 3);
    w.save_episode("t").unwrap();
    let plain_root = w.finalize().unwrap();
    let plain_info: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(plain_root.join("meta/info.json")).unwrap())
            .unwrap();
    assert_eq!(plain_info["video_path"], serde_json::Value::Null);
}

fn add_frames_named(w: &mut DatasetWriter, name: &str, n: usize) {
    for i in 0..n {
        let s = [0.1 * i as f64, 0.2 * i as f64];
        w.add_frame(&[(name, &s)]).unwrap();
    }
}

#[test]
fn episodes_parquet_carries_the_four_video_columns() {
    let dir = tmpdir("cols");
    let lengths = [4usize, 3];
    let root = record_episodes(&dir, &lengths);
    let batch = episodes_table(&root);
    let schema = batch.schema();

    // Names AND types, hand-built: int64 indices, float64 timestamps, and
    // they come LAST (after the meta/episodes file indices) — the position
    // the pyarrow bridge appended them at.
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        &names[names.len() - 4..],
        &[
            "videos/observation.images.cam/chunk_index",
            "videos/observation.images.cam/file_index",
            "videos/observation.images.cam/from_timestamp",
            "videos/observation.images.cam/to_timestamp",
        ]
    );
    for (name, dtype) in [
        ("videos/observation.images.cam/chunk_index", DataType::Int64),
        ("videos/observation.images.cam/file_index", DataType::Int64),
        (
            "videos/observation.images.cam/from_timestamp",
            DataType::Float64,
        ),
        (
            "videos/observation.images.cam/to_timestamp",
            DataType::Float64,
        ),
    ] {
        let f = schema.field_with_name(name).unwrap();
        assert_eq!(f.data_type(), &dtype, "{name}");
        assert!(f.is_nullable(), "{name}");
    }

    let i64s = |name: &str| -> Vec<i64> {
        let c = batch.column_by_name(name).unwrap();
        let a = c.as_any().downcast_ref::<Int64Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    };
    let f64s = |name: &str| -> Vec<f64> {
        let c = batch.column_by_name(name).unwrap();
        let a = c.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    };
    assert_eq!(
        i64s("videos/observation.images.cam/chunk_index"),
        vec![0, 0]
    );
    assert_eq!(i64s("videos/observation.images.cam/file_index"), vec![0, 1]);
    assert_eq!(
        f64s("videos/observation.images.cam/from_timestamp"),
        vec![0.0, 0.0]
    );
    assert_eq!(
        f64s("videos/observation.images.cam/to_timestamp"),
        vec![
            lengths[0] as f64 / f64::from(FPS),
            lengths[1] as f64 / f64::from(FPS)
        ]
    );
    // A video feature has no per-episode stats columns (no pixels pass
    // through the writer) — exactly what the bridge left behind.
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("stats/observation.images")),
        "{names:?}"
    );
    assert!(names.contains(&"stats/observation.state/mean"));
}

#[test]
fn video_key_has_no_data_column_and_the_dataset_still_reads() {
    let dir = tmpdir("nodata");
    let root = record_episodes(&dir, &[4, 3]);

    let data = root.join("data/chunk-000/file-000.parquet");
    let file = fs::File::open(&data).unwrap();
    let schema = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .schema()
        .clone();
    assert!(
        schema.field_with_name(KEY).is_err(),
        "a dtype-video key must not exist in the data parquet: {:?}",
        schema.fields()
    );

    // The reader ignores video features (their frames are not its business)
    // and still resolves every episode.
    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 2);
    assert_eq!(r.info().features[KEY].dtype, "video");
    let ep = r.read_episode(1).unwrap();
    assert_eq!(ep.len(), 3);
    assert_eq!(ep.features["observation.state"].len(), 3);
    assert!(ep.images.is_empty());
}

#[test]
fn edit_ops_still_refuse_a_video_dataset() {
    // The rewrite would have to renumber and move the mp4s too, which it does
    // not do — so it must refuse BEFORE touching anything, not silently drop
    // the camera. (Growing the writer's video support does not change this.)
    let dir = tmpdir("edit");
    let root = record_episodes(&dir, &[4, 3]);
    let err = caliper_dataset::edit::delete_episodes(&root, &[0])
        .unwrap_err()
        .to_string();
    assert!(err.contains("video"), "{err}");
    // Untouched: same episode count, both mp4s still there.
    let r = DatasetReader::open(&root).unwrap();
    assert_eq!(r.total_episodes(), 2);
    assert!(
        dir.join(format_video_path(DEFAULT_VIDEO_PATH, KEY, 0, 1).unwrap())
            .is_file()
    );
}

#[test]
fn the_dataset_doctor_reads_a_video_dataset_without_inventing_findings() {
    // Rust decodes no mp4s, so the video key is simply outside the doctor's
    // scope — what it must NOT do is report the video feature as missing
    // stats (D002) or choke on the extra episode columns.
    let dir = tmpdir("doctor");
    let root = record_episodes(&dir, &[24, 24]);
    let report = caliper_dataset::analyze(&root, caliper_dataset::AnalyzeOptions::default())
        .expect("the doctor must open a dtype-video dataset");
    assert_eq!(report.total_episodes, 2);
    let about_video: Vec<&str> = report
        .findings
        .iter()
        .filter(|f| f.feature.as_deref() == Some(KEY))
        .map(|f| f.code.as_str())
        .collect();
    assert!(about_video.is_empty(), "{about_video:?}");
    assert!(!report.features.contains_key(KEY));
}

#[test]
fn stats_json_nests_video_pixels_like_image_features() {
    let dir = tmpdir("stats");
    let root = record_episodes(&dir, &[4, 3]);
    let stats: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("meta/stats.json")).unwrap()).unwrap();
    assert_eq!(
        stats[KEY],
        serde_json::json!({
            "min": [[[0.0]], [[0.1]], [[0.2]]],
            "max": [[[1.0]], [[0.9]], [[0.8]]],
            "mean": [[[0.5]], [[0.4]], [[0.3]]],
            "std": [[[0.25]], [[0.2]], [[0.15]]],
            "count": [7],
        })
    );
    // The vector feature's own aggregation is untouched by the merge.
    assert!(stats["observation.state"]["mean"].is_array());
    assert!(stats["observation.state"]["mean"][0].is_number());
}

#[test]
fn multi_episode_chunk_arithmetic_passes_through() {
    // Three episodes at the caller's own chunk rollover (chunks_size 2 on the
    // video side): (0,0), (0,1), (1,0) — the writer stores what it is given,
    // it does not re-derive the layout.
    let dir = tmpdir("chunks");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    let slots = [(0u64, 0u64), (0, 1), (1, 0)];
    for &(chunk, file) in &slots {
        add_frames(&mut w, 5);
        touch_video(&dir, chunk, file);
        w.register_episode_video(KEY, chunk, file, 0.0, 5.0 / f64::from(FPS))
            .unwrap();
        w.save_episode("t").unwrap();
    }
    w.set_video_stats(KEY, pixel_stats()).unwrap();
    let root = w.finalize().unwrap();

    let batch = episodes_table(&root);
    let col = |name: &str| -> Vec<i64> {
        let c = batch.column_by_name(name).unwrap();
        let a = c.as_any().downcast_ref::<Int64Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    };
    assert_eq!(
        col("videos/observation.images.cam/chunk_index"),
        vec![0, 0, 1]
    );
    assert_eq!(
        col("videos/observation.images.cam/file_index"),
        vec![0, 1, 0]
    );
    assert!(dir.join("videos").join(KEY).join("chunk-001").is_dir());
}

#[test]
fn several_episodes_may_share_one_file() {
    // lerobot's own writer concatenates episodes into one mp4 up to
    // `video_files_size_in_mb`; consecutive spans of the same file are legal
    // and must survive verbatim.
    let dir = tmpdir("shared");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    touch_video(&dir, 0, 0);
    let mut from = 0.0;
    for n in [4usize, 6] {
        add_frames(&mut w, n);
        let to = from + n as f64 / f64::from(FPS);
        w.register_episode_video(KEY, 0, 0, from, to).unwrap();
        w.save_episode("t").unwrap();
        from = to;
    }
    w.set_video_stats(KEY, pixel_stats()).unwrap();
    let root = w.finalize().unwrap();
    let batch = episodes_table(&root);
    let c = batch
        .column_by_name("videos/observation.images.cam/from_timestamp")
        .unwrap();
    let a = c.as_any().downcast_ref::<Float64Array>().unwrap();
    assert_eq!(a.value(0), 0.0);
    assert_eq!(a.value(1), 4.0 / f64::from(FPS));
}

#[test]
fn save_episode_requires_every_video_registered() {
    let dir = tmpdir("missing_reg");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    add_frames(&mut w, 4);
    let err = w.save_episode("t").unwrap_err().to_string();
    assert!(err.contains("no registration"), "{err}");
    assert!(err.contains(KEY), "{err}");
    // The frames are still buffered — the refusal loses nothing.
    assert_eq!(w.buffered_frames(), 4);
    touch_video(&dir, 0, 0);
    w.register_episode_video(KEY, 0, 0, 0.0, 4.0 / f64::from(FPS))
        .unwrap();
    w.save_episode("t").unwrap();
    assert_eq!(w.total_episodes(), 1);
}

#[test]
fn registration_is_validated() {
    let dir = tmpdir("reg_valid");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    let e = |r: Result<(), caliper_dataset::Error>| r.unwrap_err().to_string();

    assert!(
        e(w.register_episode_video("nope", 0, 0, 0.0, 1.0)).contains("not a declared video"),
        "unknown key must be refused"
    );
    assert!(
        e(w.register_episode_video("observation.state", 0, 0, 0.0, 1.0))
            .contains("not a declared video")
    );
    assert!(e(w.register_episode_video(KEY, 0, 0, f64::NAN, 1.0)).contains("finite"));
    assert!(e(w.register_episode_video(KEY, 0, 0, 1.0, 1.0)).contains("from_timestamp"));
    assert!(e(w.register_episode_video(KEY, 0, 0, -1.0, 1.0)).contains("from_timestamp"));

    w.register_episode_video(KEY, 0, 0, 0.0, 4.0 / f64::from(FPS))
        .unwrap();
    assert!(
        e(w.register_episode_video(KEY, 0, 1, 0.0, 1.0)).contains("already registered"),
        "one slot per feature per episode"
    );
}

#[test]
fn registration_refuses_a_backwards_or_overlapping_layout() {
    let dir = tmpdir("reg_order");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    add_frames(&mut w, 4);
    w.register_episode_video(KEY, 1, 5, 0.0, 4.0 / f64::from(FPS))
        .unwrap();
    w.save_episode("t").unwrap();

    add_frames(&mut w, 4);
    let err = w
        .register_episode_video(KEY, 1, 4, 0.0, 4.0 / f64::from(FPS))
        .unwrap_err()
        .to_string();
    assert!(err.contains("must advance"), "{err}");
    let err = w
        .register_episode_video(KEY, 1, 5, 0.0, 4.0 / f64::from(FPS))
        .unwrap_err()
        .to_string();
    assert!(err.contains("overlap"), "{err}");
    // Same file, starting where the previous episode ended: fine.
    w.register_episode_video(KEY, 1, 5, 4.0 / f64::from(FPS), 8.0 / f64::from(FPS))
        .unwrap();
}

#[test]
fn the_registered_span_must_cover_the_episode() {
    let dir = tmpdir("span");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    add_frames(&mut w, 10);
    // 9 frames' worth of video for a 10-frame episode — the desync the
    // bridge caught by comparing frame counts.
    w.register_episode_video(KEY, 0, 0, 0.0, 9.0 / f64::from(FPS))
        .unwrap();
    let err = w.save_episode("t").unwrap_err().to_string();
    assert!(err.contains("desync"), "{err}");
    assert!(err.contains("10"), "{err}");
}

#[test]
fn discard_buffered_drops_registrations_too() {
    let dir = tmpdir("discard");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    add_frames(&mut w, 4);
    w.register_episode_video(KEY, 0, 0, 0.0, 4.0 / f64::from(FPS))
        .unwrap();
    w.discard_buffered();
    assert_eq!(w.buffered_frames(), 0);

    // The abandoned take's slot is gone: the next episode must register its
    // own (and may reuse the file the discarded one claimed).
    add_frames(&mut w, 6);
    let err = w.save_episode("t").unwrap_err().to_string();
    assert!(err.contains("no registration"), "{err}");
    touch_video(&dir, 0, 0);
    w.register_episode_video(KEY, 0, 0, 0.0, 6.0 / f64::from(FPS))
        .unwrap();
    w.save_episode("t").unwrap();
    w.set_video_stats(KEY, pixel_stats()).unwrap();
    let root = w.finalize().unwrap();
    let batch = episodes_table(&root);
    assert_eq!(batch.num_rows(), 1);
}

#[test]
fn set_video_stats_is_validated() {
    let dir = tmpdir("stats_valid");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    let e = |r: Result<(), caliper_dataset::Error>| r.unwrap_err().to_string();
    assert!(e(w.set_video_stats("nope", pixel_stats())).contains("not a declared video"));

    let mut short = pixel_stats();
    short.mean = vec![0.5, 0.4];
    assert!(e(w.set_video_stats(KEY, short)).contains("2 entries"));

    let mut nan = pixel_stats();
    nan.std = vec![0.1, f64::NAN, 0.2];
    assert!(e(w.set_video_stats(KEY, nan)).contains("finite"));

    let mut counts = pixel_stats();
    counts.count = vec![1, 2];
    assert!(e(w.set_video_stats(KEY, counts)).contains("count"));
}

#[test]
fn finalize_requires_stats_for_every_video_feature() {
    let dir = tmpdir("no_stats");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    add_frames(&mut w, 4);
    touch_video(&dir, 0, 0);
    w.register_episode_video(KEY, 0, 0, 0.0, 4.0 / f64::from(FPS))
        .unwrap();
    w.save_episode("t").unwrap();
    let err = w.finalize().unwrap_err().to_string();
    assert!(err.contains("no pixel stats"), "{err}");
    // The recorded frames still landed with a valid parquet footer.
    let data = dir.join("data/chunk-000/file-000.parquet");
    let file = fs::File::open(&data).unwrap();
    assert_eq!(
        ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .metadata()
            .file_metadata()
            .num_rows(),
        4
    );
}

#[test]
fn finalize_refuses_a_registration_whose_episode_was_never_saved() {
    let dir = tmpdir("orphan");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    add_frames(&mut w, 4);
    touch_video(&dir, 0, 0);
    w.register_episode_video(KEY, 0, 0, 0.0, 4.0 / f64::from(FPS))
        .unwrap();
    w.save_episode("t").unwrap();
    // A second episode's video is encoded and registered, but no frames are
    // ever saved for it — that mp4 would belong to nothing.
    touch_video(&dir, 0, 1);
    w.register_episode_video(KEY, 0, 1, 0.0, 4.0 / f64::from(FPS))
        .unwrap();
    w.set_video_stats(KEY, pixel_stats()).unwrap();
    let err = w.finalize().unwrap_err().to_string();
    assert!(err.contains("never saved"), "{err}");
}

#[test]
fn finalize_requires_every_registered_mp4_on_disk() {
    let dir = tmpdir("no_mp4");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    add_frames(&mut w, 4);
    w.register_episode_video(KEY, 0, 7, 0.0, 4.0 / f64::from(FPS))
        .unwrap();
    w.save_episode("t").unwrap();
    w.set_video_stats(KEY, pixel_stats()).unwrap();
    let err = w.finalize().unwrap_err().to_string();
    assert!(err.contains("file-007.mp4"), "{err}");
    assert!(err.contains("does not exist"), "{err}");
    // Nothing was registered in meta/ — no half-written video dataset.
    assert!(!dir.join("meta/info.json").exists());
}

#[test]
fn video_features_reject_frame_payloads_and_bad_specs() {
    let dir = tmpdir("payload");
    let mut w = DatasetWriter::create(&dir, spec()).unwrap();
    let s = [0.0, 0.0];
    let err = w
        .add_frame(&[("observation.state", &s), (KEY, &s)])
        .unwrap_err()
        .to_string();
    assert!(err.contains("video feature"), "{err}");
    let err = w
        .add_frame_with_images(&[("observation.state", &s)], &[(KEY, b"png")])
        .unwrap_err()
        .to_string();
    assert!(err.contains("video feature"), "{err}");

    for (bad, want) in [
        (FeatureSpec::video("v", 0, W, 3, "av1", "yuv420p"), "height"),
        (FeatureSpec::video("v", H, W, 1, "av1", "yuv420p"), "RGB"),
        (FeatureSpec::video("v", H, W, 3, "", "yuv420p"), "non-empty"),
        (FeatureSpec::video("v", H, W, 3, "av1", " "), "non-empty"),
    ] {
        let err = DatasetWriter::create(tmpdir("badspec"), DatasetSpec::new(FPS, "bot", vec![bad]))
            .err()
            .expect("bad video spec must be refused")
            .to_string();
        assert!(err.contains(want), "{err}");
    }
}

#[test]
fn two_cameras_as_video_beside_one_as_image() {
    // The realistic multi-camera dataset: a wrist camera kept as PNG frames in
    // the data parquet and two views as mp4s. It is the case that stresses the
    // per-frame payload arithmetic (video keys count towards NEITHER the vector
    // nor the image list) and the per-key column loop at once.
    const LEFT: &str = "observation.images.left";
    const RIGHT: &str = "observation.images.right";
    const WRIST: &str = "observation.images.wrist";
    const IH: usize = 4;
    const IW: usize = 3;

    let dir = tmpdir("two_cams");
    let mut w = DatasetWriter::create(
        &dir,
        DatasetSpec::new(
            FPS,
            "cam_bot",
            vec![
                FeatureSpec::vector("observation.state", 2, None),
                FeatureSpec::video(LEFT, H, W, 3, "av1", "yuv420p"),
                FeatureSpec::image(WRIST, IH, IW, 3),
                // Declared LAST and with a different codec: the columns must
                // follow declaration order, and each key keeps its own facts.
                FeatureSpec::video(RIGHT, H / 2, W / 2, 3, "h264", "yuv420p"),
            ],
        ),
    )
    .unwrap();

    let png = {
        let px: Vec<u8> = (0..IH * IW * 3).map(|i| (i % 256) as u8).collect();
        let mut out = Vec::new();
        let mut enc = png::Encoder::new(&mut out, IW as u32, IH as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().unwrap();
        wr.write_image_data(&px).unwrap();
        wr.finish().unwrap();
        out
    };
    let touch = |key: &str, chunk: u64, file: u64| {
        let path = dir.join(format_video_path(DEFAULT_VIDEO_PATH, key, chunk, file).unwrap());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"not really an mp4").unwrap();
    };

    let lengths = [4usize, 6];
    for (ep, &n) in lengths.iter().enumerate() {
        for i in 0..n {
            let s = [0.1 * i as f64, 0.2 * i as f64];
            // One vector payload, one image payload — the two video keys take
            // none, and passing one for them is what the kind checks refuse.
            w.add_frame_with_images(&[("observation.state", &s)], &[(WRIST, &png)])
                .unwrap();
        }
        let to = n as f64 / f64::from(FPS);
        // Registering only ONE of the two videos must not let the episode close,
        // and the refusal must name the one still missing.
        touch(LEFT, 0, ep as u64);
        w.register_episode_video(LEFT, 0, ep as u64, 0.0, to)
            .unwrap();
        let err = w.save_episode("t").unwrap_err().to_string();
        assert!(
            err.contains("no registration") && err.contains(RIGHT),
            "{err}"
        );
        // The two cameras may sit at completely different places in the layout.
        touch(RIGHT, ep as u64, 0);
        w.register_episode_video(RIGHT, ep as u64, 0, 0.0, to)
            .unwrap();
        w.save_episode("t").unwrap();
    }
    // Both video features need their own stats; one missing is still a refusal.
    w.set_video_stats(LEFT, pixel_stats()).unwrap();
    let err = w.finalize().unwrap_err().to_string();
    assert!(
        err.contains("no pixel stats") && err.contains(RIGHT),
        "{err}"
    );

    // finalize() is one-shot, so re-record the same dataset with both stats in.
    let dir = tmpdir("two_cams_ok");
    let mut w = DatasetWriter::create(
        &dir,
        DatasetSpec::new(
            FPS,
            "cam_bot",
            vec![
                FeatureSpec::vector("observation.state", 2, None),
                FeatureSpec::video(LEFT, H, W, 3, "av1", "yuv420p"),
                FeatureSpec::image(WRIST, IH, IW, 3),
                FeatureSpec::video(RIGHT, H / 2, W / 2, 3, "h264", "yuv420p"),
            ],
        ),
    )
    .unwrap();
    let touch = |key: &str, chunk: u64, file: u64| {
        let path = dir.join(format_video_path(DEFAULT_VIDEO_PATH, key, chunk, file).unwrap());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"not really an mp4").unwrap();
    };
    for (ep, &n) in lengths.iter().enumerate() {
        for i in 0..n {
            let s = [0.1 * i as f64, 0.2 * i as f64];
            w.add_frame_with_images(&[("observation.state", &s)], &[(WRIST, &png)])
                .unwrap();
        }
        let to = n as f64 / f64::from(FPS);
        touch(LEFT, 0, ep as u64);
        touch(RIGHT, ep as u64, 0);
        w.register_episode_video(LEFT, 0, ep as u64, 0.0, to)
            .unwrap();
        w.register_episode_video(RIGHT, ep as u64, 0, 0.0, to)
            .unwrap();
        w.save_episode("t").unwrap();
    }
    w.set_video_stats(LEFT, pixel_stats()).unwrap();
    let mut right_stats = pixel_stats();
    right_stats.count = vec![10];
    w.set_video_stats(RIGHT, right_stats).unwrap();
    let root = w.finalize().unwrap();

    // Eight video columns, grouped per key in DECLARATION order (left, right),
    // after the image feature's own per-episode stats columns.
    let batch = episodes_table(&root);
    let schema = batch.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let video_cols: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("videos/"))
        .collect();
    assert_eq!(
        video_cols,
        [
            "videos/observation.images.left/chunk_index",
            "videos/observation.images.left/file_index",
            "videos/observation.images.left/from_timestamp",
            "videos/observation.images.left/to_timestamp",
            "videos/observation.images.right/chunk_index",
            "videos/observation.images.right/file_index",
            "videos/observation.images.right/from_timestamp",
            "videos/observation.images.right/to_timestamp",
        ]
    );
    let i64s = |name: &str| -> Vec<i64> {
        let c = batch.column_by_name(name).unwrap();
        let a = c.as_any().downcast_ref::<Int64Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    };
    // The two cameras' layouts stay independent — no key inherits the other's.
    assert_eq!(
        i64s("videos/observation.images.left/file_index"),
        vec![0, 1]
    );
    assert_eq!(
        i64s("videos/observation.images.left/chunk_index"),
        vec![0, 0]
    );
    assert_eq!(
        i64s("videos/observation.images.right/file_index"),
        vec![0, 0]
    );
    assert_eq!(
        i64s("videos/observation.images.right/chunk_index"),
        vec![0, 1]
    );
    // Only the IMAGE camera folds per-episode stats; neither video key does.
    assert!(names.contains(&"stats/observation.images.wrist/mean"));
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("stats/observation.images.left"))
    );
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("stats/observation.images.right"))
    );

    // The data parquet holds the image column and NEITHER video key.
    let file = fs::File::open(root.join("data/chunk-000/file-000.parquet")).unwrap();
    let data_schema = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .schema()
        .clone();
    assert!(data_schema.field_with_name(WRIST).is_ok());
    assert!(data_schema.field_with_name(LEFT).is_err());
    assert!(data_schema.field_with_name(RIGHT).is_err());

    // info.json: three camera entries, two dtypes, per-key container facts.
    let info: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("meta/info.json")).unwrap()).unwrap();
    assert_eq!(info["features"][WRIST]["dtype"], "image");
    assert_eq!(info["features"][LEFT]["dtype"], "video");
    assert_eq!(info["features"][RIGHT]["dtype"], "video");
    assert_eq!(info["features"][LEFT]["info"]["video.codec"], "av1");
    assert_eq!(info["features"][RIGHT]["info"]["video.codec"], "h264");
    assert_eq!(
        info["features"][RIGHT]["info"]["video.height"],
        (H / 2) as u64
    );
    assert_eq!(
        info["features"][RIGHT]["shape"],
        serde_json::json!([H / 2, W / 2, 3])
    );

    // stats.json: both video keys nested like the image key, counts intact.
    let stats: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("meta/stats.json")).unwrap()).unwrap();
    assert_eq!(stats[LEFT]["count"], serde_json::json!([7]));
    assert_eq!(stats[RIGHT]["count"], serde_json::json!([10]));
    assert_eq!(
        stats[LEFT]["mean"],
        serde_json::json!([[[0.5]], [[0.4]], [[0.3]]])
    );
    assert!(stats[WRIST]["mean"][0][0][0].is_number());
}
